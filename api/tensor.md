# torch.Tensor

## Tensor

A `torch.Tensor` is a multi-dimensional matrix containing numbers.

## Initializing Tensors

A tensor can be constructed from a Python `list` or sequence using the `torch.tensor()` constructor:

```python repl
torch.tensor([[1., -1.], [1., -1.]])
torch.tensor([1, 2, 3, 4])
```

You can also use tensor creation ops:

```python repl
torch.zeros(2, 3)
torch.ones(2, 3)
torch.randn(2, 3)
```

## Tensor Attributes

| Attribute                                       | Description                                                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [`shape`]({torch.Tensor.shape})                 | Returns the shape of the tensor.                                                                                              |
| [`data`]({torch.Tensor.data})                   | Returns a detached view of the tensor data (no gradient).                                                                     |
| [`requires_grad`]({torch.Tensor.requires_grad}) | Is `True` if gradients need to be computed for this Tensor.                                                                   |
| [`grad`]({torch.Tensor.grad})                   | This attribute is `None` by default and becomes a Tensor the first time a call to `backward()` computes gradients for `self`. |
| [`T`]({torch.Tensor.T})                         | Returns a view of this tensor with its dimensions reversed.                                                                   |

## Tensor Methods

The contents of a tensor can be accessed using Python’s indexing and slicing notation:

```python repl
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x[0]
x[1, 2]
x[0:2]
```

Integer indexing selects along dimension 0 and returns a tensor with one fewer dimension. Tuple indexing chains dimension-0 selections (e.g. `x[i, j]` is equivalent to `x[i][j]`). Slice indexing returns a new tensor containing the selected rows.

Use `torch.Tensor.item()` to get a Python number from a tensor containing a single value:

```python repl
x = torch.tensor([[1]])
x
x.item()
x = torch.tensor(2.5)
x
x.item()
```

| Method                                                       | Description                                                                                        |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| [`tolist()`]({torch.Tensor.tolist})                          | Return tensor data as a (nested) Python list, or a Python scalar for 0-d tensors.                  |
| [`item()`]({torch.Tensor.item})                              | Returns the value of this tensor as a standard Python number.                                      |
| [`size(dim=None)`]({torch.Tensor.size})                      | Returns the size of the `self` tensor.                                                             |
| [`dim()`]({torch.Tensor.dim})                                | Returns the number of dimensions of `self` tensor.                                                 |
| [`numel()`]({torch.Tensor.numel})                            | Returns the total number of elements in the `self` tensor.                                         |
| [`backward(gradient=None)`]({torch.Tensor.backward})         | Computes the gradient of current tensor w.r.t. graph leaves.                                       |
| [`detach()`]({torch.Tensor.detach})                          | Returns a new Tensor, detached from the current graph.                                             |
| [`zero_()`]({torch.Tensor.zero_})                            | Fills `self` tensor with zeros.                                                                    |
| [`retain_grad()`]({torch.Tensor.retain_grad})                | Enables .grad attribute for non-leaf Tensors.                                                      |
| [`reshape(*args)`]({torch.Tensor.reshape})                   | Returns a tensor with the same data and number of elements as `self` but with the specified shape. |
| [`view(*args)`]({torch.Tensor.view})                         | Returns a new tensor with the same data as the `self` tensor but of a different shape.             |
| [`squeeze(dim=None)`]({torch.Tensor.squeeze})                | Returns a tensor with all specified dimensions of input of size 1 removed.                         |
| [`unsqueeze(dim)`]({torch.Tensor.unsqueeze})                 | Returns a new tensor with a dimension of size one inserted at the specified position.              |
| [`expand(*args)`]({torch.Tensor.expand})                     | Returns a new view of the `self` tensor with singleton dimensions expanded to a larger size.       |
| [`transpose(dim0, dim1)`]({torch.Tensor.transpose})          | Returns a tensor that is a transposed version of `self`.                                           |
| [`flatten(start_dim=0, end_dim=-1)`]({torch.Tensor.flatten}) | Flattens `self` tensor by reshaping it into a one-dimensional tensor.                              |

### Reductions

| Method                                                 | Description                                              |
| ------------------------------------------------------ | -------------------------------------------------------- |
| [`sum(dim=None, keepdim=False)`]({torch.Tensor.sum})   | Returns the sum of all elements in the tensor.           |
| [`mean(dim=None, keepdim=False)`]({torch.Tensor.mean}) | Returns the mean value of all elements in the tensor.    |
| [`max(dim=None, keepdim=False)`]({torch.Tensor.max})   | Returns the maximum value of all elements in the tensor. |
| [`min(dim=None, keepdim=False)`]({torch.Tensor.min})   | Returns the minimum value of all elements in the tensor. |

### Math Operations

Many math operations are exposed directly as tensor methods:
[`add(other)`]({torch.Tensor.add}), [`sub(other)`]({torch.Tensor.sub}), [`mul(other)`]({torch.Tensor.mul}), [`div(other)`]({torch.Tensor.div}), [`pow(other)`]({torch.Tensor.pow}), [`matmul(other)`]({torch.Tensor.matmul}), [`neg()`]({torch.Tensor.neg}), [`abs()`]({torch.Tensor.abs}), [`log()`]({torch.Tensor.log}), [`exp()`]({torch.Tensor.exp}), [`sqrt()`]({torch.Tensor.sqrt}), [`square()`]({torch.Tensor.square}), [`sin()`]({torch.Tensor.sin}), [`cos()`]({torch.Tensor.cos}), [`tan()`]({torch.Tensor.tan}), [`sigmoid()`]({torch.Tensor.sigmoid}), [`relu()`]({torch.Tensor.relu}), [`sign()`]({torch.Tensor.sign}), [`reciprocal()`]({torch.Tensor.reciprocal}), [`nan_to_num()`]({torch.Tensor.nan_to_num}).

Additionally, the Python `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`, and `__matmul__` magic methods are implemented to match standard math operators (+, -, \*, /, \*\*, @).

### Comparison Operations

| Method                                                                              | Description                                                       |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`lt(other)`]({torch.Tensor.lt})                                                    | Computes `self < other` element-wise.                             |
| [`gt(other)`]({torch.Tensor.gt})                                                    | Computes `self > other` element-wise.                             |
| [`le(other)`]({torch.Tensor.le})                                                    | Computes `self <= other` element-wise.                            |
| [`ge(other)`]({torch.Tensor.ge})                                                    | Computes `self >= other` element-wise.                            |
| [`eq(other)`]({torch.Tensor.eq})                                                    | Computes `self == other` element-wise.                            |
| [`ne(other)`]({torch.Tensor.ne})                                                    | Computes `self != other` element-wise.                            |
| [`allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`]({torch.Tensor.allclose}) | This function checks if `self` and `other` satisfy the condition. |

---

## Method and Attribute Details

### [[torch.Tensor.shape]]

Returns the shape of the tensor as a tuple.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.shape
```

### torch.Tensor.data

Returns a detached view of the tensor data (no gradient).

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
x.data
x.data.requires_grad
```

### [[torch.Tensor.requires_grad]]

Is `True` if gradients need to be computed for this Tensor, `False` otherwise.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
x.requires_grad
```

### [[torch.Tensor.grad]]

This attribute is `None` by default and becomes a Tensor the first time a call to `backward()` computes gradients for `self`.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
y = x.sum()
y.backward()
x.grad
```

### torch.Tensor.T

Returns a view of this tensor with its dimensions reversed.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.T
```

> [!WARNING]
> The use of `Tensor.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error.

### [[torch.Tensor.tolist]]

```python
Tensor.tolist() -> list
```

Return tensor data as a (nested) Python list, or a Python scalar for 0-d tensors.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.tolist()
```

### [[torch.Tensor.item]]

```python
Tensor.item() -> number
```

Returns the value of this tensor as a standard Python number. This only works for tensors with one element.

```python repl
x = torch.tensor([3])
x.item()
```

### [[torch.Tensor.size]]

```python
Tensor.size(dim=None) -> tuple or int
```

Returns the size of the `self` tensor. If `dim` is not specified, returns the full shape tuple.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.size()
x.size(1)
```

### [[torch.Tensor.dim]]

```python
Tensor.dim() -> int
```

Returns the number of dimensions of `self` tensor.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.dim()
```

### [[torch.Tensor.numel]]

```python
Tensor.numel() -> int
```

Returns the total number of elements in the `self` tensor.

```python repl
x = torch.tensor([[1, 2], [3, 4]])
x.numel()
```

### [[torch.Tensor.backward]]

```python
Tensor.backward(gradient=None)
```

Computes the gradient of current tensor w.r.t. graph leaves.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
y = x.sum()
y.backward()
x.grad
```

### [[torch.Tensor.detach]]

```python
Tensor.detach() -> Tensor
```

Returns a new Tensor, detached from the current graph.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
y = x.detach()
y.requires_grad
```

### [[torch.Tensor.zero_]]

```python
Tensor.zero_() -> Tensor
```

Fills `self` tensor with zeros.

```python repl
x = torch.tensor([1., 2.])
x.zero_()
x
```

### [[torch.Tensor.retain_grad]]

```python
Tensor.retain_grad()
```

Enables .grad attribute for non-leaf Tensors.

```python repl
x = torch.tensor([1., 2.], requires_grad=True)
y = x * 2
y.retain_grad()
z = y.sum()
z.backward()
y.grad
```

### [[torch.Tensor.reshape]]

```python
Tensor.reshape(*args) -> Tensor
```

See []({torch.reshape}).

### [[torch.Tensor.view]]

```python
Tensor.view(*args) -> Tensor
```

Returns a new tensor with the same data as the `self` tensor but of a different shape.

```python repl
x = torch.tensor([1, 2, 3, 4])
x.view(2, 2)
```

### [[torch.Tensor.squeeze]]

```python
Tensor.squeeze(dim=None) -> Tensor
```

See []({torch.squeeze}).

### [[torch.Tensor.unsqueeze]]

```python
Tensor.unsqueeze(dim) -> Tensor
```

See []({torch.unsqueeze}).

### [[torch.Tensor.expand]]

```python
Tensor.expand(*args) -> Tensor
```

Returns a new view of the `self` tensor with singleton dimensions expanded to a larger size.

```python repl
x = torch.tensor([1])
x.expand(3)
```

### [[torch.Tensor.transpose]]

```python
Tensor.transpose(dim0, dim1) -> Tensor
```

See []({torch.transpose}).

### [[torch.Tensor.flatten]]

```python
Tensor.flatten(start_dim=0, end_dim=-1) -> Tensor
```

See []({torch.flatten}).

### [[torch.Tensor.sum]]

```python
Tensor.sum(dim=None, keepdim=False) -> Tensor
```

See []({torch.sum}).

### [[torch.Tensor.mean]]

```python
Tensor.mean(dim=None, keepdim=False) -> Tensor
```

See []({torch.mean}).

### [[torch.Tensor.max]]

```python
Tensor.max(dim=None, keepdim=False) -> Tensor
```

See []({torch.max}).

### [[torch.Tensor.min]]

```python
Tensor.min(dim=None, keepdim=False) -> Tensor
```

See []({torch.min}).

### [[torch.Tensor.add]]

```python
Tensor.add(other) -> Tensor
```

See []({torch.add}).

### [[torch.Tensor.sub]]

```python
Tensor.sub(other) -> Tensor
```

See []({torch.sub}).

### [[torch.Tensor.mul]]

```python
Tensor.mul(other) -> Tensor
```

See []({torch.mul}).

### [[torch.Tensor.div]]

```python
Tensor.div(other) -> Tensor
```

See []({torch.div}).

### [[torch.Tensor.pow]]

```python
Tensor.pow(other) -> Tensor
```

See []({torch.pow}).

### [[torch.Tensor.matmul]]

```python
Tensor.matmul(other) -> Tensor
```

See []({torch.matmul}).

### [[torch.Tensor.neg]]

```python
Tensor.neg() -> Tensor
```

See []({torch.neg}).

### [[torch.Tensor.abs]]

```python
Tensor.abs() -> Tensor
```

See []({torch.abs}).

### [[torch.Tensor.log]]

```python
Tensor.log() -> Tensor
```

See []({torch.log}).

### [[torch.Tensor.exp]]

```python
Tensor.exp() -> Tensor
```

See []({torch.exp}).

### [[torch.Tensor.sqrt]]

```python
Tensor.sqrt() -> Tensor
```

See []({torch.sqrt}).

### [[torch.Tensor.square]]

```python
Tensor.square() -> Tensor
```

See []({torch.square}).

### [[torch.Tensor.sin]]

```python
Tensor.sin() -> Tensor
```

See []({torch.sin}).

### [[torch.Tensor.cos]]

```python
Tensor.cos() -> Tensor
```

See []({torch.cos}).

### [[torch.Tensor.tan]]

```python
Tensor.tan() -> Tensor
```

See []({torch.tan}).

### [[torch.Tensor.sigmoid]]

```python
Tensor.sigmoid() -> Tensor
```

See []({torch.sigmoid}).

### [[torch.Tensor.relu]]

```python
Tensor.relu() -> Tensor
```

See []({torch.relu}).

### [[torch.Tensor.sign]]

```python
Tensor.sign() -> Tensor
```

See []({torch.sign}).

### [[torch.Tensor.reciprocal]]

```python
Tensor.reciprocal() -> Tensor
```

See []({torch.reciprocal}).

### [[torch.Tensor.nan_to_num]]

```python
Tensor.nan_to_num() -> Tensor
```

See []({torch.nan_to_num}).

### [[torch.Tensor.lt]]

```python
Tensor.lt(other) -> Tensor
```

See []({torch.lt}).

### [[torch.Tensor.gt]]

```python
Tensor.gt(other) -> Tensor
```

See []({torch.gt}).

### [[torch.Tensor.le]]

```python
Tensor.le(other) -> Tensor
```

See []({torch.le}).

### [[torch.Tensor.ge]]

```python
Tensor.ge(other) -> Tensor
```

See []({torch.ge}).

### [[torch.Tensor.eq]]

```python
Tensor.eq(other) -> Tensor
```

See []({torch.eq}).

### [[torch.Tensor.ne]]

```python
Tensor.ne(other) -> Tensor
```

See []({torch.ne}).

### [[torch.Tensor.allclose]]

```python
Tensor.allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool
```

See []({torch.allclose}).
