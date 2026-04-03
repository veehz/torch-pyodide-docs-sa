# torch

Top-level functions in the `torch` namespace.

## Contents

### Tensors

| Function                           | Description                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`is_tensor`]({torch.is_tensor})   | Returns True if obj is a PyTorch tensor.                                                                |
| [`is_nonzero`]({torch.is_nonzero}) | Returns True if the input is a single element tensor which is not equal to zero after type conversions. |
| [`numel`]({torch.numel})           | Returns the total number of elements in the input tensor.                                               |

#### Creation Ops

| Function                           | Description                                                                                                                                  |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [`tensor`]({torch.Tensor})         | Create a tensor from data.                                                                                                                   |
| [`zeros`]({torch.zeros})           | Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.                                       |
| [`zeros_like`]({torch.zeros_like}) | Returns a tensor filled with the scalar value 0, with the same shape as input                                                                |
| [`ones`]({torch.ones})             | Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.                                       |
| [`ones_like`]({torch.ones_like})   | Returns a tensor filled with the scalar value 1, with the same shape as input                                                                |
| [`empty`]({torch.empty})           | Returns an uninitialized tensor.                                                                                                             |
| [`empty_like`]({torch.empty_like}) | Returns an uninitialized tensor with the same shape as input                                                                                 |
| [`full`]({torch.full})             | Returns a tensor filled with the scalar value, with the shape defined by the variable argument size.                                         |
| [`full_like`]({torch.full_like})   | Returns a tensor filled with the scalar value, with the same shape as input                                                                  |
| [`arange`]({torch.arange})         | Returns a 1-D tensor of size $$ \lceil \frac{\text{end} - \text{start}}{\text{step}} \rceil $$ with values from start to end with step step. |
| [`linspace`]({torch.linspace})     | Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `end`, inclusive.                            |

### Random sampling

| Function                               | Description                                                                                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`seed`]({torch.seed})                 | Sets the seed for generating random numbers to a non-deterministic random number.                                                                         |
| [`manual_seed`]({torch.manual_seed})   | Sets the seed for generating random numbers.                                                                                                              |
| [`rand`]({torch.rand})                 | Returns a tensor filled with random numbers from a uniform distribution on the interval $$ [0, 1) $$.                                                     |
| [`rand_like`]({torch.rand_like})       | Returns a tensor with the same size as `input` that is filled with random numbers from a uniform distribution on the interval $$ [0, 1) $$.               |
| [`randint`]({torch.randint})           | Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive).                                        |
| [`randint_like`]({torch.randint_like}) | Returns a tensor with the same shape as `input` that is filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive). |
| [`randn`]({torch.randn})               | Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).         |
| [`randn_like`]({torch.randn_like})     | Returns a tensor with the same size as `input` that is filled with random numbers from a normal distribution with mean 0 and variance 1.                  |
| [`randperm`]({torch.randperm})         | Returns a tensor containing a random permutation of integers in the interval $$ [0, n) $$.                                                                |

### Locally disabling gradient computation

The context manager `torch.no_grad()` is helpful for locally disabling and enabling gradient computation.

```python repl
x = torch.zeros(1, requires_grad=True)
with torch.no_grad():
    y = x * 2
y.requires_grad
```

| Function                                     | Description                                         |
| -------------------------------------------- | --------------------------------------------------- |
| [`no_grad`]({torch.no_grad})                 | Context manager that disables gradient computation. |
| [`is_grad_enabled`]({torch.is_grad_enabled}) | Returns True if grad mode is currently enabled.     |

### Math operations

| Function                               | Description                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| [`abs`]({torch.abs})                   | Computes the absolute value of each element in `input`.                        |
| [`neg`]({torch.neg})                   | Returns a new tensor with the negative of the elements of `input`.             |
| [`sign`]({torch.sign})                 | Returns a new tensor with the sign of the elements of `input`.                 |
| [`reciprocal`]({torch.reciprocal})     | Returns a new tensor with the reciprocal of the elements of `input`.           |
| [`nan_to_num`]({torch.nan_to_num})     | Replaces NaN, positive infinity, and negative infinity values in `input`.      |
| [`add`]({torch.add})                   | Adds `other` to `input`.                                                       |
| [`sub`]({torch.sub})                   | Subtracts `other` from `input`.                                                |
| [`mul`]({torch.mul})                   | Multiplies `input` by `other`.                                                 |
| [`div`]({torch.div})                   | Divides `input` by `other`.                                                    |
| [`pow`]({torch.pow})                   | Takes the power of each element in `input` with `other`.                       |
| [`fmod`]({torch.fmod})                 | Computes the element-wise remainder of the division of `input` by `other`.     |
| [`square`]({torch.square})             | Returns a new tensor with the square of the elements of `input`.               |
| [`sqrt`]({torch.sqrt})                 | Returns a new tensor with the square-root of the elements of `input`.          |
| [`exp`]({torch.exp})                   | Returns a new tensor with the exponential of the elements of the input tensor. |
| [`log`]({torch.log})                   | Returns a new tensor with the natural logarithm of the elements of `input`.    |
| [`sin`]({torch.sin})                   | Returns a new tensor with the sine of the elements of `input`.                 |
| [`cos`]({torch.cos})                   | Returns a new tensor with the cosine of the elements of `input`.               |
| [`tan`]({torch.tan})                   | Returns a new tensor with the tangent of the elements of `input`.              |
| [`matmul`]({torch.matmul})             | Matrix product of two tensors.                                                 |
| [`sum`]({torch.sum})                   | Returns the sum of all elements in the `input` tensor.                         |
| [`mean`]({torch.mean})                 | Returns the mean value of all elements in the `input` tensor.                  |
| [`max`]({torch.max})                   | Returns the maximum value of all elements in the `input` tensor.               |
| [`min`]({torch.min})                   | Returns the minimum value of all elements in the `input` tensor.               |
| [`maximum`]({torch.maximum})           | Computes the element-wise maximum of `input` and `other`.                      |
| [`minimum`]({torch.minimum})           | Computes the element-wise minimum of `input` and `other`.                      |
| [`softmax`]({torch.softmax})           | Applies the softmax function along a given dimension.                          |
| [`clamp`]({torch.clamp})               | Clamps all elements in `input` into the range `[min, max]`.                    |

### Comparison operations

| Function                               | Description                                                       |
| -------------------------------------- | ----------------------------------------------------------------- |
| [`lt`]({torch.lt})                     | Computes `input < other` element-wise.                            |
| [`gt`]({torch.gt})                     | Computes `input > other` element-wise.                            |
| [`le`]({torch.le})                     | Computes `input <= other` element-wise.                           |
| [`ge`]({torch.ge})                     | Computes `input >= other` element-wise.                           |
| [`eq`]({torch.eq})                     | Computes `input == other` element-wise.                           |
| [`ne`]({torch.ne})                     | Computes `input != other` element-wise.                           |
| [`allclose`]({torch.allclose})         | Checks if all elements of `input` and `other` are close.         |

## Operations

### [[torch.reshape]]

```python
torch.reshape(input, *args) -> Tensor
```

Returns a tensor with the same data and number of elements as `input` but with the specified shape.

```python repl
x = torch.tensor([1, 2, 3, 4])
torch.reshape(x, (2, 2))
```

### [[torch.squeeze]]

```python
torch.squeeze(input, dim=None) -> Tensor
```

Returns a tensor with all specified dimensions of input of size 1 removed.

```python repl
x = torch.tensor([[[1], [2]]])
torch.squeeze(x)
```

### [[torch.unsqueeze]]

```python
torch.unsqueeze(input, dim) -> Tensor
```

Returns a new tensor with a dimension of size one inserted at the specified position.

```python repl
x = torch.tensor([1, 2])
torch.unsqueeze(x, 0)
```

### [[torch.transpose]]

```python
torch.transpose(input, dim0, dim1) -> Tensor
```

Returns a tensor that is a transposed version of `input`.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.transpose(x, 0, 1)
```

### [[torch.flatten]]

```python
torch.flatten(input, start_dim=0, end_dim=-1) -> Tensor
```

Flattens `input` tensor by reshaping it into a one-dimensional tensor.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.flatten(x)
```

### [[torch.sum]]

```python
torch.sum(input, dim=None, keepdim=False) -> Tensor
```

Returns the sum of all elements in the tensor.

```python repl
x = torch.tensor([1., 2.])
torch.sum(x)
```

### [[torch.mean]]

```python
torch.mean(input, dim=None, keepdim=False) -> Tensor
```

Returns the mean value of all elements in the tensor.

```python repl
x = torch.tensor([1., 2.])
torch.mean(x)
```

### [[torch.max]]

```python
torch.max(input, dim=None, keepdim=False) -> Tensor
```

Returns the maximum value of all elements in the tensor.

```python repl
x = torch.tensor([1., 2.])
torch.max(x)
```

### [[torch.min]]

```python
torch.min(input, dim=None, keepdim=False) -> Tensor
```

Returns the minimum value of all elements in the tensor.

```python repl
x = torch.tensor([1., 2.])
torch.min(x)
```

### [[torch.add]]

```python
torch.add(input, other) -> Tensor
```

Adds `other` to `input`.

```python repl
x = torch.tensor([1, 2])
torch.add(x, 3)
```

### [[torch.sub]]

```python
torch.sub(input, other) -> Tensor
```

Subtracts `other` from `input`.

```python repl
x = torch.tensor([1, 2])
torch.sub(x, 3)
```

### [[torch.mul]]

```python
torch.mul(input, other) -> Tensor
```

Multiplies `input` by `other`.

```python repl
x = torch.tensor([1, 2])
torch.mul(x, 3)
```

### [[torch.div]]

```python
torch.div(input, other) -> Tensor
```

Divides `input` by `other`.

```python repl
x = torch.tensor([1., 2.])
torch.div(x, 2)
```

### [[torch.pow]]

```python
torch.pow(input, other) -> Tensor
```

Takes the power of each element in `input` with `other`.

```python repl
x = torch.tensor([2., 3.])
torch.pow(x, 2)
```

### [[torch.matmul]]

```python
torch.matmul(input, other) -> Tensor
```

Matrix product of two tensors.

```python repl
x = torch.tensor([[1, 2]])
y = torch.tensor([[3], [4]])
torch.matmul(x, y)
```

### [[torch.neg]]

```python
torch.neg(input) -> Tensor
```

Returns a new tensor with the negative of the elements of `input`.

```python repl
x = torch.tensor([1, -2])
torch.neg(x)
```

### [[torch.abs]]

```python
torch.abs(input) -> Tensor
```

Computes the element-wise absolute value of the given input tensor.

```python repl
x = torch.tensor([1, -2])
torch.abs(x)
```

### [[torch.log]]

```python
torch.log(input) -> Tensor
```

Returns a new tensor with the natural logarithm of the elements of `input`.

```python repl
x = torch.tensor([1., 2.])
torch.log(x)
```

### [[torch.exp]]

```python
torch.exp(input) -> Tensor
```

Returns a new tensor with the exponential of the elements of the input tensor.

```python repl
x = torch.tensor([1., 2.])
torch.exp(x)
```

### [[torch.sqrt]]

```python
torch.sqrt(input) -> Tensor
```

Returns a new tensor with the square-root of the elements of `input`.

```python repl
x = torch.tensor([1., 4.])
torch.sqrt(x)
```

### [[torch.square]]

```python
torch.square(input) -> Tensor
```

Returns a new tensor with the square of the elements of `input`.

```python repl
x = torch.tensor([1., 4.])
torch.square(x)
```

### [[torch.sin]]

```python
torch.sin(input) -> Tensor
```

Returns a new tensor with the sine of the elements of `input`.

```python repl
x = torch.tensor([0., 3.14])
torch.sin(x)
```

### [[torch.cos]]

```python
torch.cos(input) -> Tensor
```

Returns a new tensor with the cosine of the elements of `input`.

```python repl
x = torch.tensor([0., 3.14])
torch.cos(x)
```

### [[torch.tan]]

```python
torch.tan(input) -> Tensor
```

Returns a new tensor with the tangent of the elements of `input`.

```python repl
x = torch.tensor([0., 3.14])
torch.tan(x)
```

### [[torch.sigmoid:torch.nn.functional.sigmoid]]

```python
torch.sigmoid(input) -> Tensor
```

Applies the sigmoid function element-wise.

```python repl
x = torch.tensor([0., 1.])
torch.sigmoid(x)
```

### [[torch.relu:torch.nn.functional.relu]]

```python
torch.relu(input) -> Tensor
```

Applies the rectified linear unit function element-wise: `max(0, x)`.

```python repl
x = torch.tensor([-1., 0., 1.])
torch.relu(x)
```

### [[torch.softmax]]

```python
torch.softmax(input, dim) -> Tensor
```

Applies the softmax function along dimension `dim`. Each slice along `dim` sums to 1. Uses the numerically stable formulation (subtracts the max before exponentiating).

**Parameters**

| Name    | Type     | Description                              |
| ------- | -------- | ---------------------------------------- |
| `input` | `Tensor` | Input tensor.                            |
| `dim`   | `int`    | Dimension along which softmax is applied. |

```python repl
x = torch.tensor([1., 2., 3.])
torch.softmax(x, dim=0)
```

### [[torch.clamp]]

```python
torch.clamp(input, min, max) -> Tensor
```

Clamps all elements in `input` into the range `[min, max]`. `torch.clip` is an alias.

**Parameters**

| Name    | Type     | Description         |
| ------- | -------- | ------------------- |
| `input` | `Tensor` | Input tensor.       |
| `min`   | `float`  | Lower bound.        |
| `max`   | `float`  | Upper bound.        |

```python repl
x = torch.tensor([-2., 0., 2., 5.])
torch.clamp(x, 0, 3)
```

### [[torch.sign]]

```python
torch.sign(input) -> Tensor
```

Returns a new tensor with the sign of the elements of `input`.

```python repl
x = torch.tensor([-1., 0., 1.])
torch.sign(x)
```

### [[torch.reciprocal]]

```python
torch.reciprocal(input) -> Tensor
```

Returns a new tensor with the reciprocal of the elements of `input`.

```python repl
x = torch.tensor([2., 4.])
torch.reciprocal(x)
```

### [[torch.nan_to_num]]

```python
torch.nan_to_num(input) -> Tensor
```

Replaces NaN, positive infinity, and negative infinity values in `input` with the corresponding replacement values.

```python repl
x = torch.tensor([1., float('nan')])
torch.nan_to_num(x)
```

### [[torch.lt]]

```python
torch.lt(input, other) -> Tensor
```

Computes `self < other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.lt(x, 2)
```

### [[torch.gt]]

```python
torch.gt(input, other) -> Tensor
```

Computes `self > other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.gt(x, 1)
```

### [[torch.le]]

```python
torch.le(input, other) -> Tensor
```

Computes `self <= other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.le(x, 1)
```

### [[torch.ge]]

```python
torch.ge(input, other) -> Tensor
```

Computes `self >= other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.ge(x, 2)
```

### [[torch.eq]]

```python
torch.eq(input, other) -> Tensor
```

Computes `self == other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.eq(x, 2)
```

### [[torch.ne]]

```python
torch.ne(input, other) -> Tensor
```

Computes `self != other` element-wise.

```python repl
x = torch.tensor([1, 2])
torch.ne(x, 2)
```

### [[torch.maximum]]

```python
torch.maximum(input, other) -> Tensor
```

Computes the element-wise maximum of `input` and `other`.

```python repl
x = torch.tensor([1., 3.])
y = torch.tensor([2., 2.])
torch.maximum(x, y)
```

### [[torch.minimum]]

```python
torch.minimum(input, other) -> Tensor
```

Computes the element-wise minimum of `input` and `other`.

```python repl
x = torch.tensor([1., 3.])
y = torch.tensor([2., 2.])
torch.minimum(x, y)
```

### [[torch.fmod]]

```python
torch.fmod(input, other) -> Tensor
```

Computes the element-wise remainder of the division of `input` by `other`.

```python repl
x = torch.tensor([7., 10.])
torch.fmod(x, 3)
```

### [[torch.allclose]]

```python
torch.allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool
```

This function checks if `input` and `other` satisfy the condition.

```python repl
x = torch.tensor([1., 2.])
y = torch.tensor([1.00001, 2.])
torch.allclose(x, y)
```

---

## Tensor Utilities

### [[torch.is_tensor]]

```python
torch.is_tensor(obj) -> bool
```

Returns `True` if `obj` is a PyTorch tensor.

```python repl
x = torch.tensor([1.])
torch.is_tensor(x)
torch.is_tensor([1.])
```

### [[torch.is_nonzero]]

```python
torch.is_nonzero(input) -> bool
```

Returns `True` if the input is a single element tensor which is not equal to zero after type conversions. Raises an error if the tensor has more than one element.

```python repl
torch.is_nonzero(torch.tensor([0.]))
torch.is_nonzero(torch.tensor([1.5]))
```

### [[torch.numel]]

```python
torch.numel(input) -> int
```

Returns the total number of elements in the `input` tensor.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.numel(x)
```

---

## Creation Ops

### [[torch.tensor]]

```python
torch.tensor(data, requires_grad=False) -> Tensor
```

Creates a tensor from `data` (a Python list or nested list of numbers).

```python repl
torch.tensor([[1., -1.], [1., -1.]])
torch.tensor([1, 2, 3, 4])
torch.tensor([1., 2.], requires_grad=True)
```

### [[torch.zeros]]

```python
torch.zeros(*size) -> Tensor
```

Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument `size`.

```python repl
torch.zeros(2, 3)
torch.zeros([2, 3])
```

### [[torch.zeros_like]]

```python
torch.zeros_like(input) -> Tensor
```

Returns a tensor filled with the scalar value 0, with the same shape as `input`.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.zeros_like(x)
```

### [[torch.ones]]

```python
torch.ones(*size) -> Tensor
```

Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument `size`.

```python repl
torch.ones(2, 3)
```

### [[torch.ones_like]]

```python
torch.ones_like(input) -> Tensor
```

Returns a tensor filled with the scalar value 1, with the same shape as `input`.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.ones_like(x)
```

### [[torch.empty]]

```python
torch.empty(*size) -> Tensor
```

Returns an uninitialized tensor with the shape defined by the variable argument `size`.

```python repl
torch.empty(2, 3)
```

### [[torch.empty_like]]

```python
torch.empty_like(input) -> Tensor
```

Returns an uninitialized tensor with the same shape as `input`.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.empty_like(x)
```

### [[torch.full]]

```python
torch.full(shape, fill_value) -> Tensor
```

Returns a tensor filled with `fill_value`, with the shape defined by `shape`.

```python repl
torch.full([2, 3], 3.14)
```

### [[torch.full_like]]

```python
torch.full_like(input, fill_value) -> Tensor
```

Returns a tensor filled with `fill_value`, with the same shape as `input`.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
torch.full_like(x, 7)
```

### [[torch.arange]]

```python
torch.arange(start, end, step=1) -> Tensor
```

Returns a 1-D tensor with values from `start` to `end` (exclusive) with step `step`.

```python repl
torch.arange(0, 5)
torch.arange(1, 5, 0.5)
```

### [[torch.linspace]]

```python
torch.linspace(start, end, steps) -> Tensor
```

Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `end`, inclusive.

```python repl
torch.linspace(0, 1, 5)
```

---

## Random Sampling

### [[torch.seed]]

```python
torch.seed()
```

Sets the seed for generating random numbers to a non-deterministic random number.

### [[torch.manual_seed]]

```python
torch.manual_seed(seed)
```

Sets the seed for generating random numbers. Use this to get reproducible results.

```python repl
torch.manual_seed(42)
torch.randn(2, 2)
```

### [[torch.rand]]

```python
torch.rand(*size) -> Tensor
```

Returns a tensor filled with random numbers from a uniform distribution on the interval $[0, 1)$.

```python repl
torch.rand(2, 3)
```

### [[torch.rand_like]]

```python
torch.rand_like(input) -> Tensor
```

Returns a tensor with the same size as `input` filled with random numbers from a uniform distribution on the interval $[0, 1)$.

```python repl
x = torch.zeros(2, 3)
torch.rand_like(x)
```

### [[torch.randint]]

```python
torch.randint(low, high, size) -> Tensor
```

Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive).

```python repl
torch.randint(0, 10, [2, 3])
```

### [[torch.randint_like]]

```python
torch.randint_like(input, low, high) -> Tensor
```

Returns a tensor with the same shape as `input` filled with random integers between `low` (inclusive) and `high` (exclusive).

```python repl
x = torch.zeros(2, 3)
torch.randint_like(x, 0, 10)
```

### [[torch.randn]]

```python
torch.randn(*size) -> Tensor
```

Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1.

```python repl
torch.randn(2, 3)
```

### [[torch.randn_like]]

```python
torch.randn_like(input) -> Tensor
```

Returns a tensor with the same size as `input` filled with random numbers from a normal distribution with mean 0 and variance 1.

```python repl
x = torch.zeros(2, 3)
torch.randn_like(x)
```

### [[torch.randperm]]

```python
torch.randperm(n) -> Tensor
```

Returns a tensor containing a random permutation of integers from $0$ to $n - 1$.

```python repl
torch.randperm(5)
```

---

## Gradient Control

### [[torch.no_grad]]

```python
torch.no_grad()
```

Context manager that disables gradient computation. Operations inside the block do not track gradients, which reduces memory usage and speeds up computation during inference.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
with torch.no_grad():
    y = x * 2
y.requires_grad
```

### [[torch.is_grad_enabled]]

```python
torch.is_grad_enabled() -> bool
```

Returns `True` if grad mode is currently enabled.

```python repl
torch.is_grad_enabled()
```
