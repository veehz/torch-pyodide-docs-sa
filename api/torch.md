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
| [`no_grad`]({torch.no.grad})                 | Context manager that disables gradient computation. |
| [`is_grad_enabled`]({torch.is.grad.enabled}) | Returns True if grad mode is currently enabled.     |

### Math operations

| Function                       | Description                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------ |
| [`abs`]({torch.abs})           | Computes the absolute value of each element in `input`.                        |
| [`add`]({torch.add})           | Adds `other` to `input`.                                                       |
| [`sub`]({torch.sub})           | Subtracts `other` from `input`.                                                |
| [`mul`]({torch.mul})           | Multiplies `input` by `other`.                                                 |
| [`div`]({torch.div})           | Divides `input` by `other`.                                                    |
| [`pow`]({torch.pow})           | Takes the power of each element in `input` with `other`.                       |
| [`matmul`]({torch.matmul})     | Matrix product of two tensors.                                                 |
| [`sum`]({torch.sum})           | Returns the sum of all elements in the `input` tensor.                         |
| [`mean`]({torch.mean})         | Returns the mean value of all elements in the `input` tensor.                  |
| [`max`]({torch.max})           | Returns the maximum value of all elements in the `input` tensor.               |
| [`min`]({torch.min})           | Returns the minimum value of all elements in the `input` tensor.               |
| [`sin`]({torch.sin})           | Returns a new tensor with the sine of the elements of `input`.                 |
| [`cos`]({torch.cos})           | Returns a new tensor with the cosine of the elements of `input`.               |
| [`tan`]({torch.tan})           | Returns a new tensor with the tangent of the elements of `input`.              |
| [`exp`]({torch.exp})           | Returns a new tensor with the exponential of the elements of the input tensor. |
| [`log`]({torch.log})           | Returns a new tensor with the natural logarithm of the elements of `input`.    |
| [`sqrt`]({torch.sqrt})         | Returns a new tensor with the square-root of the elements of `input`.          |
| [`maximum`]({torch.maximum})   | Computes the element-wise maximum of `input` and `other`.                      |
| [`minimum`]({torch.minimum})   | Computes the element-wise minimum of `input` and `other`.                      |
| [`allclose`]({torch.allclose}) | This function checks if all `input` and `other` satisfy the condition.         |
| [`cat`]({torch.cat})           | Concatenates the given sequence of seq tensors in the given dimension.         |

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

### torch.sigmoid

```python
torch.sigmoid(input) -> Tensor
```

Applies the sigmoid function element-wise.

```python repl
x = torch.tensor([0., 1.])
torch.sigmoid(x)
```

### torch.relu

```python
torch.relu(input) -> Tensor
```

Applies the rectified linear unit function element-wise: `max(0, x)`.

```python repl
x = torch.tensor([-1., 0., 1.])
torch.relu(x)
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
