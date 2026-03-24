# torch.nn

Building blocks for computational graphs and neural networks.

## Base Classes

### [[torch.nn.Module]]

All neural network components should subclass `nn.Module`.

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
```

**Key Methods**

| Method                                         | Description                                   |
| ---------------------------------------------- | --------------------------------------------- |
| [`forward(x)`]({torch.nn.Module.forward})      | Define the computation. Called by `model(x)`. |
| [`parameters()`]({torch.nn.Module.parameters}) | Returns a list of all learnable parameters.   |
| [`zero_grad()`]({torch.nn.Module.zero_grad})   | Zeros gradients of all parameters.            |

---

### [[torch.nn.parameter.Parameter]]

A kind of Tensor that is to be considered a module parameter. Also accessible as `torch.nn.Parameter`.

```python
nn.Parameter(data, requires_grad=True)
```

**Parameters**

| Name            | Type     | Description                                                 |
| --------------- | -------- | ----------------------------------------------------------- |
| `data`          | `Tensor` | The tensor data to wrap as a parameter.                     |
| `requires_grad` | `bool`   | Whether the parameter requires a gradient. Default: `True`. |

**Notes**

- Parameters always default to `requires_grad=True`. Unlike plain tensors, you do not need to pass `requires_grad=True` explicitly.
- `torch.no_grad()` does **not** affect Parameter creation. A Parameter constructed inside a `no_grad` block still has `requires_grad=True`.

**Example**

```python
w = nn.Parameter(torch.randn(3, 4))
print(w.requires_grad)

with torch.no_grad():
    p = nn.Parameter(torch.randn(3, 4))
    print(p.requires_grad)
```

---

### [[torch.nn.Sequential]]

A sequential container. Modules will be added to it in the order they are passed in the constructor.

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

---

## Layers

### [[torch.nn.Linear]]

```python
nn.Linear(in_features, out_features, bias=True)
```

Applies a linear transformation: `y = xW^T + b`.

**Parameters**

| Name           | Type   | Description                                         |
| -------------- | ------ | --------------------------------------------------- |
| `in_features`  | `int`  | Size of each input sample.                          |
| `out_features` | `int`  | Size of each output sample.                         |
| `bias`         | `bool` | If `False`, no bias term is added. Default: `True`. |

**Attributes**

| Attribute | Description                                                     |
| --------- | --------------------------------------------------------------- |
| `.weight` | Learnable weight tensor of shape `(out_features, in_features)`. |
| `.bias`   | Learnable bias tensor of shape `(out_features,)`.               |

**Example**

```python
layer = nn.Linear(4, 2)
x = torch.randn(3, 4)  # batch of 3
out = layer(x)          # shape: (3, 2)
```

---

### [[torch.nn.Conv1d]]

```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 1D convolution over an input signal composed of several input planes.

**Parameters**

| Name           | Type            | Description                                                                |
| -------------- | --------------- | -------------------------------------------------------------------------- |
| `in_channels`  | `int`           | Number of channels in the input.                                           |
| `out_channels` | `int`           | Number of channels produced by the convolution.                            |
| `kernel_size`  | `int` or `list` | Size of the convolving kernel.                                             |
| `stride`       | `int` or `list` | Stride of the convolution. Default: `1`.                                   |
| `padding`      | `int` or `list` | Zero-padding added to both sides of the input. Default: `0`.               |
| `dilation`     | `int` or `list` | Spacing between kernel elements. Default: `1`.                             |
| `groups`       | `int`           | Number of blocked connections from input to output channels. Default: `1`. |
| `bias`         | `bool`          | If `True`, adds a learnable bias to the output. Default: `True`.           |

---

### [[torch.nn.Conv2d]]

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 2D convolution over an input signal composed of several input planes.

**Parameters**

| Name           | Type            | Description                                                                |
| -------------- | --------------- | -------------------------------------------------------------------------- |
| `in_channels`  | `int`           | Number of channels in the input image.                                     |
| `out_channels` | `int`           | Number of channels produced by the convolution.                            |
| `kernel_size`  | `int` or `list` | Size of the convolving kernel.                                             |
| `stride`       | `int` or `list` | Stride of the convolution. Default: `1`.                                   |
| `padding`      | `int` or `list` | Zero-padding added to both sides of the input. Default: `0`.               |
| `dilation`     | `int` or `list` | Spacing between kernel elements. Default: `1`.                             |
| `groups`       | `int`           | Number of blocked connections from input to output channels. Default: `1`. |
| `bias`         | `bool`          | If `True`, adds a learnable bias to the output. Default: `True`.           |

---

### [[torch.nn.Conv3d]]

```python
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 3D convolution over an input signal composed of several input planes.

**Parameters**

| Name           | Type            | Description                                                                |
| -------------- | --------------- | -------------------------------------------------------------------------- |
| `in_channels`  | `int`           | Number of channels in the input volume.                                    |
| `out_channels` | `int`           | Number of channels produced by the convolution.                            |
| `kernel_size`  | `int` or `list` | Size of the convolving kernel.                                             |
| `stride`       | `int` or `list` | Stride of the convolution. Default: `1`.                                   |
| `padding`      | `int` or `list` | Zero-padding added to both sides of the input. Default: `0`.               |
| `dilation`     | `int` or `list` | Spacing between kernel elements. Default: `1`.                             |
| `groups`       | `int`           | Number of blocked connections from input to output channels. Default: `1`. |
| `bias`         | `bool`          | If `True`, adds a learnable bias to the output. Default: `True`.           |

---

## Loss Functions

### [[torch.nn.MSELoss]]

```python
nn.MSELoss()
```

Mean Squared Error loss: `loss = mean((pred - target)^2)`.

**Example**

```python
criterion = nn.MSELoss()
pred = torch.tensor([1.0, 2.0])
target = torch.tensor([1.5, 2.5])
loss = criterion(pred, target)  # scalar tensor
```

---

### [[torch.nn.L1Loss]]

```python
nn.L1Loss()
```

Creates a criterion that measures the mean absolute error (MAE) between each element in the input $x$ and target $y$.

---

### [[torch.nn.BCELoss]]

```python
nn.BCELoss()
```

Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.

---

### [[torch.nn.CrossEntropyLoss]]

```python
nn.CrossEntropyLoss()
```

This criterion computes the cross entropy loss between input logits and target class indices.

**Input:** `(N, C)` where `N` is the batch size and `C` is the number of classes. Values are unnormalized logits.

**Target:** `(N,)` where each value is an integer in `[0, C)`.

**Example**

```python
criterion = nn.CrossEntropyLoss()
input = torch.randn(3, 5)      # batch of 3, 5 classes
target = torch.tensor([1, 0, 4])
loss = criterion(input, target)
```

---

## Activation Functions

### [[torch.nn.ReLU]]

```python
nn.ReLU()
```

Applies the rectified linear unit function element-wise: `f(x) = max(0, x)`.

---

### [[torch.nn.Sigmoid]]

```python
nn.Sigmoid()
```

Applies the sigmoid function: `f(x) = 1 / (1 + e^(-x))`.

---

## Functional API

Functional interfaces are exposed under `torch.nn.functional`.

### [[torch.nn.functional.relu]]

Applies the rectified linear unit function element-wise.

### [[torch.nn.functional.sigmoid]]

Applies the sigmoid function element-wise.

### [[torch.nn.functional.cross_entropy]]

Computes the cross entropy loss between input logits and target.

### [[torch.nn.functional.conv1d]]

Applies a 1D convolution over an input signal.

### [[torch.nn.functional.conv2d]]

Applies a 2D convolution over an input signal.

### [[torch.nn.functional.conv3d]]

Applies a 3D convolution over an input signal.
