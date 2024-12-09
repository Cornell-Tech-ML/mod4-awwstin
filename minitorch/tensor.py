"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_ops import TensorBackend
from .fast_ops import FastOps

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        if backend is None:
            backend = TensorBackend(ops=FastOps)
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set whether the tensor requires gradient computation.

        Args:
        ----
            x (bool): If True, gradients will be computed for this tensor.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation.

        Returns
        -------
            bool: True if the tensor requires gradient, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Get an item from the tensor."""
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Set an item in the tensor."""
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        """Set the backend for the tensor."""
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Create a new tensor from existing tensor data."""
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data."""
        if backend is None:
            backend = TensorBackend(ops=FastOps)
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def expand_to_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Expand this tensor to the desired shape.

        Args:
        ----
            shape (tuple of int): The desired expanded size.

        Returns:
        -------
            Tensor: A new tensor with expanded size.

        """
        # Case 1: Both the same shape.
        if self.shape == shape:
            return self

        # Implement the logic to expand self to the new shape.
        true_shape = TensorData.shape_broadcast(self.shape, shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(self, buf)
        return buf

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Reverse subtraction: (scalar) - Tensor.

        Args:
        ----
            other (int, float, or Tensor): The left operand.

        Returns:
        -------
            Tensor: Result of the subtraction.

        """
        if not isinstance(other, Tensor):
            other = self.__class__.from_scalar(other, backend=self.backend)
        return other - self

    def unsqueeze(self, dim: int) -> "Tensor":
        """Return a new Tensor with a dimension of size one inserted at the specified position.

        Args:
        ----
            dim (int): The index at which to insert the singleton dimension.

        Returns:
        -------
            Tensor: A new Tensor with the singleton dimension inserted.

        """
        # Handle negative dimensions
        if dim < 0:
            dim += len(self.shape) + 1  # +1 because we're adding a new dimension

        # Check if the dimension is within the valid range
        if not (0 <= dim <= len(self.shape)):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-len(self.shape) - 1}, {len(self.shape)}], but got {dim})"
            )

        # Compute the new shape by inserting 1 at the specified dimension
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        # Compute the new strides by inserting a stride corresponding to the new dimension
        # For simplicity, you can set the stride for the new dimension to the product of the strides
        # Alternatively, you may need to adjust this based on your implementation of TensorData
        new_strides = list(self._tensor.strides)
        if dim < len(new_strides):
            new_strides.insert(dim, new_strides[dim] * new_shape[dim + 1])
        else:
            new_strides.append(1)  # For the last dimension

        # Create a new TensorData with the new shape and strides
        new_tensor_data = TensorData(
            self._tensor._storage, tuple(new_shape), tuple(new_strides)
        )

        # Return a new Tensor with the updated TensorData and the same backend
        return Tensor(new_tensor_data, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor. If None, uses self.shape.

        Returns:
        -------
            Tensor: A new tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            """Create a tensor filled with zeros of a given shape."""
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    @classmethod
    def from_scalar(
        cls, value: Union[int, float], backend: Optional[TensorBackend] = None
    ) -> Tensor:
        """Create a Tensor from a scalar value.

        Args:
        ----
            value (int or float): The scalar value.
            backend (TensorBackend, optional): The backend to use.

        Returns:
        -------
            Tensor: A new Tensor instance representing the scalar.

        """
        if backend is None:
            backend = TensorBackend(ops=FastOps)
        data = TensorData(np.array([value]), (1,), (1,))
        return cls(data, backend=backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is a constant (has no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the inputs that created this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients.

        Args:
        ----
            d_output: The gradient of the output.

        Returns:
        -------
            An iterable of (Variable, gradient) tuples.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute gradients of this tensor w.r.t. graph leaves.

        Args:
        ----
            grad_output (Optional[Tensor]): Gradient w.r.t. the tensor. If None, assumes a scalar tensor with gradient 1.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """True division: Tensor / Tensor."""
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Reverse true division: (scalar) / Tensor."""
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns shape of the tensor"""
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return len(self.shape)

    def __add__(self, other: TensorLike) -> Tensor:
        """Addition: Tensor + Tensor."""
        other = self._ensure_tensor(other)
        return Add.apply(self, other)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Subtraction: Tensor - Tensor."""
        other = self._ensure_tensor(other)
        return Add.apply(self, Neg.apply(other))

    def __mul__(self, other: TensorLike) -> Tensor:
        """Multiplication: Tensor * Tensor."""
        other = self._ensure_tensor(other)
        return Mul.apply(self, other)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Less than comparison: Tensor < Tensor."""
        other = self._ensure_tensor(other)
        return LT.apply(self, other)

    def __eq__(self, other: TensorLike) -> Tensor:
        """Equality comparison: Tensor == Tensor."""
        other = self._ensure_tensor(other)
        return EQ.apply(self, other)

    def __gt__(self, other: TensorLike) -> Tensor:
        """Greater than comparison: Tensor > Tensor."""
        other = self._ensure_tensor(other)
        return LT.apply(other, self)

    def __neg__(self) -> Tensor:
        """Negate the tensor."""
        return Neg.apply(self)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Reverse addition: (scalar) + Tensor."""
        return self.__add__(other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Reverse multiplication: (scalar) * Tensor."""
        return self.__mul__(other)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements are True along a given dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: Result of the 'all' operation.

        """
        if dim is not None:
            return All.apply(self, dim)
        else:
            reshaped_tensor = self.contiguous().view(int(operators.prod(self.shape)))
            return All.apply(reshaped_tensor, 0)

    def is_close(self, other: TensorLike) -> Tensor:
        """Check if all elements are close to the corresponding elements of another tensor.

        Args:
        ----
            other (TensorLike): The tensor to compare with.

        Returns:
        -------
            Tensor: Result of the 'is close' operation.

        """
        other = self._ensure_tensor(other)
        return IsClose.apply(self, other)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid function to the tensor."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU function to the tensor."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the logarithm function to the tensor."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function to the tensor."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum the elements of the tensor along a given dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: Result of the 'sum' operation.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(int(operators.prod(self.shape))), 0)
        else:
            return Sum.apply(self, dim)

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of the tensor along a given dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, reduces all dimensions.

        Returns:
        -------
            Tensor: Result of the 'mean' operation.

        """
        total_elements = self.shape[dim] if dim is not None else self.size
        return self.sum(dim) * (1.0 / total_elements)

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order (int): The new order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        return Permute.apply(self, order)

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor.

        Args:
        ----
            *shape (int): The new shape of the tensor.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        shape_list = list(map(float, shape))
        shape_tensor = Tensor.make(shape_list, (len(shape),), backend=self.backend)
        return View.apply(self, shape_tensor)

    def zero_grad_(self) -> None:
        """Reset the gradients of all parameters to None."""
        self.grad = None

    @staticmethod
    def tensor(
        data: Union[Storage, List[float], np.ndarray],
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data.

        Args:
        ----
            data: Data to create tensor from
            backend: Backend to use for tensor operations

        Returns:
        -------
            A new Tensor

        """
        if backend is None:
            backend = TensorBackend(ops=FastOps)
        if isinstance(data, np.ndarray):
            # Convert numpy array to list and get its shape
            return Tensor.make(data.flatten().tolist(), data.shape, backend=backend)
        elif isinstance(data, (list, tuple)):
            # Convert to numpy array to get proper shape, then back to list
            arr = np.array(data)
            return Tensor.make(arr.flatten().tolist(), arr.shape, backend=backend)
        else:
            # Handle scalar values
            return Tensor.from_scalar(data, backend=backend)
