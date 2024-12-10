from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, Optional


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Create copies of the input values to modify
    vals_plus = list(vals)
    vals_minus = list(vals)

    # Increment and decrement the specified argument by epsilon
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Evaluate the function at the incremented and decremented points
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Compute the central difference approximation
    derivative = (f_plus - f_minus) / (2 * epsilon)
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique id for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns an iterable of this variable's parents in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this variable's parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(v: Variable) -> None:
        """Depth-first search to find the topological order."""
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        order.append(v)

    dfs(variable)
    return reversed(order)  # We reverse to get the correct topological order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable

    Args:
    ----
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # Get the variables in topological order
    topo_order = list(topological_sort(variable))

    # Initialize derivatives dictionary
    derivatives = {var.unique_id: 0.0 for var in topo_order}
    derivatives[variable.unique_id] = deriv

    # Iterate over variables in reverse topological order
    for var in topo_order:
        d_output = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_parent in var.chain_rule(d_output):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0.0
                derivatives[parent.unique_id] += d_parent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()
    dim: Optional[int] = None

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values


@dataclass
class History:
    """Class to track the history of computations for autograd."""

    last_fn: Any
    ctx: Context
    inputs: Tuple[Any, ...]

    def __init__(self, last_fn: Any, ctx: Context, inputs: Tuple[Any, ...]) -> None:
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs
