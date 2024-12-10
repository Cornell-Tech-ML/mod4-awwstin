from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Ensure the input is contiguous for view
    input = input.contiguous()

    # Reshape into (batch, channel, new_height, kh, new_width, kw)
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)

    reshaped = reshaped.permute(0, 1, 2, 4, 3, 5)
    reshaped = reshaped.contiguous()

    # Flatten the last two dimensions (kh and kw) into a single dimension
    tiled = reshaped.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: (batch, channel, height, width)
        kernel: (kh, kw)

    Returns:
    -------
        (batch, channel, new_height, new_width)

    """
    tiled, new_h, new_w = tile(input, kernel)
    kh, kw = kernel
    batch, channel = tiled.shape[0], tiled.shape[1]

    # tiled: (batch, channel, new_h, new_w, kh*kw)
    # sum over the last dimension
    summed = tiled.sum(dim=4)  # should now have shape (batch, channel, new_h, new_w)

    # If sum did not remove the dimension, it might leave (batch, channel, new_h, new_w, 1)
    # but we know we want exactly (batch, channel, new_h, new_w), so force a view:
    summed = summed.view(batch, channel, new_h, new_w)

    avg = summed / (kh * kw)
    return avg


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along dimension dim."""
    max_vals = max_reduce(input, dim)
    mask = input == max_vals
    return mask


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max reduction.

        Args:
        ----
            ctx: Function context
            input: Input tensor
            dim: Dimension to reduce along

        Returns:
        -------
            Tensor with maximum values along specified dimension

        """
        d = int(dim.item())
        ctx.save_for_backward(input, d)
        return max_reduce(input, d)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max reduction.

        Args:
        ----
            ctx: Function context
            grad_output: Gradient from downstream operations

        Returns:
        -------
            Tuple of (gradient with respect to input, gradient with respect to dimension)

        """
        input, d = ctx.saved_values
        max_vals = max_reduce(input, d)
        mask = input == max_vals
        return mask * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to reduce along

    Returns:
    -------
        Tensor with maximum values along specified dimension

    """
    t = input._ensure_tensor(dim)
    out = Max.apply(input, t)
    return out


def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply softmax along a dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor with softmax applied

    """
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Apply log softmax along a dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to apply log softmax

    Returns:
    -------
        Tensor with log softmax applied

    """
    max_val = max(input, dim)
    logsumexp = ((input - max_val).exp().sum(dim)).log() + max_val
    return input - logsumexp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: (batch, channel, height, width)
        kernel: (kh, kw)

    Returns:
    -------
        (batch, channel, new_height, new_width)

    """
    batch, channel, _, _ = input.shape
    tiled, new_h, new_w = tile(input, kernel)
    pooled = max(tiled, 4).view(batch, channel, new_h, new_w)
    return pooled


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to input tensor.

    Args:
    ----
        input: Input tensor
        rate: Dropout rate (probability of setting values to zero)
        ignore: If True, return input unchanged (useful for inference)

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input
    mask = rand(input.shape) > rate
    return input * mask
