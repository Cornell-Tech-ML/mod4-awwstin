# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator that jits a function for CUDA device execution.

    Args:
    ----
        fn: Function to compile
        kwargs: Additional arguments for jit compiler

    Returns:
    -------
        Device-compiled function

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator that jits a function for CPU execution.

    Args:
    ----
        fn: Function to compile
        kwargs: Additional arguments for jit compiler

    Returns:
    -------
        CPU-compiled function

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a function to each element of a tensor on GPU.

        Args:
        ----
            fn: Function to apply to each element

        Returns:
        -------
            Mapped tensor function

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a function to each pair of elements from two tensors on GPU.

        Args:
        ----
            fn: Function to apply to each pair of elements

        Returns:
        -------
            Zipped tensor function

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce a tensor over a specified dimension on GPU.

        Args:
        ----
            fn: Reduction function
            start: Starting value for reduction

        Returns:
        -------
            Reduced tensor function

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Compute matrix multiplication of two tensors on GPU.

        Args:
        ----
            a (Tensor): First tensor
            b (Tensor): Second tensor

        Returns:
        -------
            Tensor: Result of matrix multiplication

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            pos_in = index_to_position(in_index, in_strides)
            pos_out = index_to_position(out_index, out_strides)
            out[pos_out] = fn(in_storage[pos_in])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            pos_a = index_to_position(a_index, a_strides)
            pos_b = index_to_position(b_index, b_strides)
            pos_out = index_to_position(out_index, out_strides)
            out[pos_out] = fn(a_storage[pos_a], b_storage[pos_b])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # Shared memory for caching values
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    # Reduce within block
    stride = 1
    while pos + stride < BLOCK_DIM:
        if pos % (2 * stride) == 0:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Write result for this block to output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum kernel to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Convert the output position to the corresponding multi-dimensional index.
        to_index(out_pos, out_shape, out_index)

        # Compute the global position in the output tensor.
        pos_g = index_to_position(out_index, out_strides)

        # Initialize cache with the reduction value.
        cache[pos] = reduce_value

        if out_pos < out_size:
            # Update the index along the reduction dimension.
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < a_shape[reduce_dim]:
                # Load the input values into shared memory.
                pos_a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[pos_a]

                # Synchronize threads to ensure all values are loaded into shared memory.
                cuda.syncthreads()

                # Perform reduction within shared memory.
                step = BLOCK_DIM // 2
                while step > 0:
                    if pos < step and pos + step < a_shape[reduce_dim]:
                        cache[pos] = fn(cache[pos], cache[step + pos])
                    # Synchronize threads before moving to the next reduction step.
                    cuda.syncthreads()
                    step //= 2

            # Write the final reduced value to the output tensor.
            if pos == 0:
                out[pos_g] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Shared memory for storing submatrices of `a` and `b`
    s_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    s_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block
    x, y = cuda.threadIdx.x, cuda.threadIdx.y

    # Loading data into shared memory
    if x < size and y < size:
        # Copy the elements of `a` and `b` to shared memory
        s_a[x, y] = a[size * x + y]
        s_b[x, y] = b[size * x + y]
        # Synchronize threads to ensure all data is loaded before computation
        cuda.syncthreads()

        # Initialize a temporary variable to accumulate the result
        temp = 0.0
        # Perform the matrix multiplication for the given row and column
        for k in range(size):
            temp += s_a[x, k] * s_b[k, y]
        # Write the computed value to the global memory
        out[x * size + y] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice square MM kernel to prepare for matmul."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    temp = 0.0
    # Loop over tiles of the shared dimension
    for start_idx in range(0, a_shape[2], BLOCK_DIM):
        # Load the tile of matrix A into shared memory
        if i < out_shape[1] and start_idx + pj < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch
                + i * a_strides[1]
                + (start_idx + pj) * a_strides[2]
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Load the tile of matrix B into shared memory
        if j < out_shape[2] and start_idx + pi < b_shape[1]:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + (start_idx + pi) * b_strides[1]
                + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Ensure all threads have finished copying to shared memory
        cuda.syncthreads()

        # Compute partial dot product for the current tile
        for k in range(BLOCK_DIM):
            if k + start_idx < a_shape[2]:
                temp += a_shared[pi, k] * b_shared[k, pj]

        # Ensure all threads have finished using the shared memory
        cuda.syncthreads()

    # Write result to global memory
    if i < out_shape[1] and j < out_shape[2]:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = temp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
