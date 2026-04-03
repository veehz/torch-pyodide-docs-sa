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

| Method                        | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| `forward(x)`                  | Define the computation. Called by `model(x)`.                 |
| `parameters()`                | Returns a list of all learnable parameters.                   |
| `named_parameters(prefix='')` | Returns a list of `(name, parameter)` tuples.                 |
| `train(mode=True)`            | Sets the module and all submodules to training mode.          |
| `eval()`                      | Sets the module and all submodules to evaluation mode.        |

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

A sequential container. Modules will be added to it in the order they are passed in the constructor. The output of each module is passed as input to the next.

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

**Methods**

| Method                  | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `append(module)`        | Appends a module to the end of the container.  |
| `insert(index, module)` | Inserts a module at the given index.           |
| `extend(sequential)`    | Appends all modules from another `Sequential`. |

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

### [[torch.nn.MaxPool2d]]

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0)
```

Applies 2D max pooling over an input signal. Accepts both 3-D `(C, H, W)` and 4-D `(N, C, H, W)` inputs.

**Parameters**

| Name          | Type            | Description                                                                                  |
| ------------- | --------------- | -------------------------------------------------------------------------------------------- |
| `kernel_size` | `int` or `list` | Size of the window to take the max over.                                                     |
| `stride`      | `int` or `list` | Stride of the window. Defaults to `kernel_size` if not provided.                             |
| `padding`     | `int` or `list` | Zero-padding added to both spatial sides. Default: `0`.                                      |

**Example**

```python
pool = nn.MaxPool2d(2)
x = torch.randn(1, 1, 4, 4)
out = pool(x)  # shape: (1, 1, 2, 2)
```

---

### [[torch.nn.Dropout]]

```python
nn.Dropout(p=0.5)
```

Randomly zeroes elements of the input tensor with probability `p` during training. During evaluation (after calling `model.eval()`), this layer is a no-op. Uses inverted dropout so that the expected sum is unchanged at train time.

**Parameters**

| Name | Type    | Description                                                   |
| ---- | ------- | ------------------------------------------------------------- |
| `p`  | `float` | Probability of an element being zeroed. Default: `0.5`.       |

**Example**

```python
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Dropout(0.5),
    nn.Linear(10, 1),
)
model.eval()   # disable dropout for inference
```

---

### [[torch.nn.Flatten]]

```python
nn.Flatten(start_dim=1, end_dim=-1)
```

Flattens a contiguous range of dimensions into a single dimension. Commonly used to transition from convolutional layers to fully-connected layers.

**Parameters**

| Name        | Type  | Description                                          |
| ----------- | ----- | ---------------------------------------------------- |
| `start_dim` | `int` | First dimension to flatten. Default: `1`.            |
| `end_dim`   | `int` | Last dimension to flatten (inclusive). Default: `-1`. |

**Example**

```python
model = nn.Sequential(
    nn.Conv2d(1, 4, 3),
    nn.Flatten(),          # (N, 4, H, W) -> (N, 4*H*W)
    nn.Linear(4 * 6 * 6, 10),
)
```

---

## Loss Functions

### [[torch.nn.MSELoss]]

```python
nn.MSELoss(reduction='mean')
```

Mean Squared Error loss: `loss = mean((pred - target)^2)`.

**Parameters**

| Name        | Type  | Default  | Description                                              |
| ----------- | ----- | -------- | -------------------------------------------------------- |
| `reduction` | `str` | `'mean'` | Specifies the reduction: `'mean'`, `'sum'`, or `'none'`. |

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
nn.L1Loss(reduction='mean')
```

Creates a criterion that measures the mean absolute error (MAE) between each element in the input $x$ and target $y$: `loss = mean(|x - y|)`.

**Parameters**

| Name        | Type  | Default  | Description                                              |
| ----------- | ----- | -------- | -------------------------------------------------------- |
| `reduction` | `str` | `'mean'` | Specifies the reduction: `'mean'`, `'sum'`, or `'none'`. |

---

### [[torch.nn.BCELoss]]

```python
nn.BCELoss(weight=None, reduction='mean')
```

Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities. Input values should be in the range $[0, 1]$ (e.g., after a sigmoid).

**Parameters**

| Name        | Type             | Default  | Description                                              |
| ----------- | ---------------- | -------- | -------------------------------------------------------- |
| `weight`    | `Tensor` or None | `None`   | A manual rescaling weight for each batch element.        |
| `reduction` | `str`            | `'mean'` | Specifies the reduction: `'mean'`, `'sum'`, or `'none'`. |

---

### [[torch.nn.CrossEntropyLoss]]

```python
nn.CrossEntropyLoss(reduction='mean')
```

This criterion computes the cross entropy loss between input logits and target class indices.

**Parameters**

| Name        | Type  | Default  | Description                                              |
| ----------- | ----- | -------- | -------------------------------------------------------- |
| `reduction` | `str` | `'mean'` | Specifies the reduction: `'mean'`, `'sum'`, or `'none'`. |

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

### [[torch.nn.NLLLoss]]

```python
nn.NLLLoss(reduction='mean')
```

Negative log-likelihood loss. The input is expected to contain **log-probabilities** for each class (typically the output of `torch.nn.functional.log_softmax`). The target is a tensor of class indices.

$$\text{loss}(x, c) = -x[c]$$

**Parameters**

| Name        | Type  | Default  | Description                                              |
| ----------- | ----- | -------- | -------------------------------------------------------- |
| `reduction` | `str` | `'mean'` | Specifies the reduction: `'mean'`, `'sum'`, or `'none'`. |

**Input:** `(N, C)` log-probabilities (e.g., output of `F.log_softmax`).

**Target:** `(N,)` class indices, each in `[0, C)`.

**Example**

```python
log_probs = torch.nn.functional.log_softmax(torch.randn(3, 5), dim=1)
target = torch.tensor([1, 0, 4])
loss = nn.NLLLoss()(log_probs, target)
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

### [[torch.nn.LeakyReLU]]

```python
nn.LeakyReLU(negative_slope=0.01)
```

Applies leaky ReLU element-wise: `f(x) = x` if `x > 0`, else `f(x) = negative_slope * x`.

**Parameters**

| Name             | Type    | Description                                          |
| ---------------- | ------- | ---------------------------------------------------- |
| `negative_slope` | `float` | Controls the angle of the negative slope. Default: `0.01`. |

---

### [[torch.nn.Softmax]]

```python
nn.Softmax(dim)
```

Applies the softmax function along `dim`. Equivalent to wrapping `torch.softmax` as a module. Useful as the final layer of a classifier.

**Parameters**

| Name  | Type  | Description                               |
| ----- | ----- | ----------------------------------------- |
| `dim` | `int` | Dimension along which softmax is applied. |

**Example**

```python
model = nn.Sequential(
    nn.Linear(4, 3),
    nn.Softmax(dim=1),
)
```

---

## Functional API

Functional interfaces are exposed under `torch.nn.functional`. Unlike the module-based API, these are stateless functions — you pass weights and inputs directly.

### [[torch.nn.functional.relu]]

```python
torch.nn.functional.relu(input) -> Tensor
```

Applies the rectified linear unit function element-wise: `f(x) = max(0, x)`.

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-1., 0., 1.])
F.relu(x)
```

### [[torch.nn.functional.sigmoid]]

```python
torch.nn.functional.sigmoid(input) -> Tensor
```

Applies the sigmoid function element-wise: `f(x) = 1 / (1 + e^(-x))`.

```python
import torch
import torch.nn.functional as F
x = torch.tensor([-1., 0., 1.])
F.sigmoid(x)
```

### [[torch.nn.functional.cross_entropy]]

```python
torch.nn.functional.cross_entropy(input, target, reduction='mean') -> Tensor
```

Computes the cross entropy loss between input logits and target class indices.

**Parameters**

| Name        | Type     | Description                                             |
| ----------- | -------- | ------------------------------------------------------- |
| `input`     | `Tensor` | Unnormalized logits of shape `(N, C)`.                  |
| `target`    | `Tensor` | Target class indices of shape `(N,)`, each in `[0, C)`. |
| `reduction` | `str`    | `'mean'` (default), `'sum'`, or `'none'`.               |

### [[torch.nn.functional.leaky_relu]]

```python
torch.nn.functional.leaky_relu(input, negative_slope=0.01) -> Tensor
```

Applies leaky ReLU element-wise.

**Parameters**

| Name             | Type     | Description                                    |
| ---------------- | -------- | ---------------------------------------------- |
| `input`          | `Tensor` | Input tensor.                                  |
| `negative_slope` | `float`  | Slope for negative values. Default: `0.01`.    |

### [[torch.nn.functional.max_pool2d]]

```python
torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0) -> Tensor
```

Applies 2D max pooling. Accepts 3-D `(C, H, W)` and 4-D `(N, C, H, W)` inputs.

**Parameters**

| Name          | Type            | Description                                                      |
| ------------- | --------------- | ---------------------------------------------------------------- |
| `input`       | `Tensor`        | Input tensor.                                                    |
| `kernel_size` | `int` or `list` | Size of the pooling window.                                      |
| `stride`      | `int` or `list` | Stride of the window. Defaults to `kernel_size` if not provided. |
| `padding`     | `int` or `list` | Zero-padding on both spatial sides. Default: `0`.                |

### [[torch.nn.functional.nll_loss]]

```python
torch.nn.functional.nll_loss(input, target, reduction='mean') -> Tensor
```

Negative log-likelihood loss. The input should be log-probabilities.

**Parameters**

| Name        | Type     | Description                                             |
| ----------- | -------- | ------------------------------------------------------- |
| `input`     | `Tensor` | Log-probabilities of shape `(N, C)`.                    |
| `target`    | `Tensor` | Class indices of shape `(N,)`, each in `[0, C)`.        |
| `reduction` | `str`    | `'mean'` (default), `'sum'`, or `'none'`.               |

### [[torch.nn.functional.conv1d]]

```python
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
```

Applies a 1D convolution over an input signal composed of several input planes.

**Parameters**

| Name       | Type             | Description                                    |
| ---------- | ---------------- | ---------------------------------------------- |
| `input`    | `Tensor`         | Input of shape `(N, C_in, L)`.                 |
| `weight`   | `Tensor`         | Filters of shape `(C_out, C_in/groups, kW)`.   |
| `bias`     | `Tensor` or None | Optional bias of shape `(C_out,)`.             |
| `stride`   | `int` or `list`  | Stride of the convolution. Default: `1`.       |
| `padding`  | `int` or `list`  | Zero-padding on both sides. Default: `0`.      |
| `dilation` | `int` or `list`  | Spacing between kernel elements. Default: `1`. |
| `groups`   | `int`            | Number of blocked connections. Default: `1`.   |

### [[torch.nn.functional.conv2d]]

```python
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
```

Applies a 2D convolution over an input signal composed of several input planes.

**Parameters**

| Name       | Type             | Description                                      |
| ---------- | ---------------- | ------------------------------------------------ |
| `input`    | `Tensor`         | Input of shape `(N, C_in, H, W)`.                |
| `weight`   | `Tensor`         | Filters of shape `(C_out, C_in/groups, kH, kW)`. |
| `bias`     | `Tensor` or None | Optional bias of shape `(C_out,)`.               |
| `stride`   | `int` or `list`  | Stride of the convolution. Default: `1`.         |
| `padding`  | `int` or `list`  | Zero-padding on both sides. Default: `0`.        |
| `dilation` | `int` or `list`  | Spacing between kernel elements. Default: `1`.   |
| `groups`   | `int`            | Number of blocked connections. Default: `1`.     |

### [[torch.nn.functional.conv3d]]

```python
torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
```

Applies a 3D convolution over an input signal composed of several input planes.

**Parameters**

| Name       | Type             | Description                                          |
| ---------- | ---------------- | ---------------------------------------------------- |
| `input`    | `Tensor`         | Input of shape `(N, C_in, D, H, W)`.                 |
| `weight`   | `Tensor`         | Filters of shape `(C_out, C_in/groups, kD, kH, kW)`. |
| `bias`     | `Tensor` or None | Optional bias of shape `(C_out,)`.                   |
| `stride`   | `int` or `list`  | Stride of the convolution. Default: `1`.             |
| `padding`  | `int` or `list`  | Zero-padding on both sides. Default: `0`.            |
| `dilation` | `int` or `list`  | Spacing between kernel elements. Default: `1`.       |
| `groups`   | `int`            | Number of blocked connections. Default: `1`.         |
