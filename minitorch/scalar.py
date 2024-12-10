from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass(frozen=True)
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: Optional[str] = None
    unique_id: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize the scalar."""
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        if self.name is None:
            object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """Return a string representation of the scalar."""
        return f"Scalar({self.data})"

    # Implementations for Task 1.2

    def __add__(self, other: ScalarLike) -> Scalar:
        """Add two scalars."""
        return Add.apply(self, other)

    def __radd__(self, other: ScalarLike) -> Scalar:
        """Add two scalars."""
        return self + other

    def __mul__(self, other: ScalarLike) -> Scalar:
        """Multiply two scalars."""
        return Mul.apply(self, other)

    def __rmul__(self, other: ScalarLike) -> Scalar:
        """Multiply two scalars."""
        return self * other

    def __sub__(self, other: ScalarLike) -> Scalar:
        """Subtract two scalars."""
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other: ScalarLike) -> Scalar:
        """Subtract two scalars."""
        return Add.apply(other, Neg.apply(self))

    def __neg__(self) -> Scalar:
        """Negate a scalar."""
        return Neg.apply(self)

    def __truediv__(self, other: ScalarLike) -> Scalar:
        """Divide two scalars."""
        return Mul.apply(self, Inv.apply(other))

    def __rtruediv__(self, other: ScalarLike) -> Scalar:
        """Divide two scalars."""
        return Mul.apply(other, Inv.apply(self))

    def __lt__(self, other: ScalarLike) -> Scalar:
        """Compare two scalars."""
        return LT.apply(self, other)

    def __gt__(self, other: ScalarLike) -> Scalar:
        """Compare two scalars."""
        return LT.apply(other, self)

    def __eq__(self, other: ScalarLike) -> Scalar:
        """Compare two scalars."""
        return EQ.apply(self, other)

    def log(self) -> Scalar:
        """Compute the natural logarithm of a scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of a scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of a scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the ReLU of a scalar."""
        return ReLU.apply(self)

    def __bool__(self) -> bool:
        """Convert a scalar to a boolean."""
        return bool(self.data)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            object.__setattr__(self, "derivative", 0.0)
        object.__setattr__(self, "derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable was created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable was created by an operation (has `last_fn`)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this scalar."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to get the derivatives of the parents."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # Use the _backward method to ensure that derivatives is a tuple
        derivatives = h.last_fn._backward(h.ctx, d_output)

        # Pair each input variable with its corresponding derivative
        # Filter out constants (variables where is_constant() is True)
        return [
            (input_var, derivative)
            for input_var, derivative in zip(h.inputs, derivatives)
            if not input_var.is_constant()
        ]

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


@staticmethod
def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f: Function from n-scalars to 1-scalar.
        *scalars: Input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
    Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
    but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
