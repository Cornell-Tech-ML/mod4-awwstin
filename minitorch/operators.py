"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, Any, Sequence, Union
from functools import reduce as py_reduce
#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return x * y


# - id
def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


# - add
def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates a number"""
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another"""
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x > 0 else 0


# - log
def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


# - log_back
def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg"""
    return d / x


# - inv
def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -d / (x * x)


# - relu_back
def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU times a second arg."""
    return d if x > 0 else 0


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(func: Callable, iterable: Iterable) -> List:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [func(x) for x in iterable]


# - zipWith
def zipWith(func: Callable, list1: Iterable, list2: Iterable) -> List:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [func(x, y) for x, y in zip(list1, list2)]


# - reduce
def reduce(func: Callable, iterable: Iterable, initializer: Any = None) -> Any:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    # Check if the iterable is empty and no initial value is provided
    if not iterable:
        if initializer is None:
            # Attempt to handle a common use case: summation
            try:
                if func(0, 0) == 0:
                    return 0
            except Exception:
                pass
            raise ValueError("reduce() of empty iterable with no initial value")
        else:
            return initializer
    return (
        py_reduce(func, iterable, initializer)
        if initializer is not None
        else py_reduce(func, iterable)
    )


#
# Use these to implement
# - negList : negate a list
def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list using map"""
    return map(lambda x: -x, lst)


# - addLists : add two lists together
def addLists(list1: List[float], list2: List[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(lambda x, y: x + y, list1, list2)


# - sum: sum lists
def sum(lst: List[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(lambda x, y: x + y, lst)


# - prod: take the product of lists
def prod(lst: Sequence[Union[int, float]]) -> float:
    """Calculate the product of all elements in a sequence using reduce"""
    return reduce(lambda x, y: x * y, lst)


def square(a: float) -> float:
    """Calculates the square of a number"""
    return a * a


def cube(a: float) -> float:
    """Calculates the cube of a number"""
    return a * a * a
