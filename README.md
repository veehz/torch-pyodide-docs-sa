# torch-pyodide-docs

Docs for [`source-academy/torch`](https://github.com/source-academy/torch) when it is implemented in Source Academy by bridging with Pyodide.

## Syntax

We introduce small plugins to support some special syntax:

- Add Link to PyTorch docs:
  - `[[torch.Tensor.shape]]`
  - Aliasing `torch.relu` to `torch.nn.functional.relu`: `[[torch.relu:torch.nn.functional.relu]]`
- Add Link to Internal docs:
  - `[shape]({torch.Tensor.shape})`
  - `[]({torch.Tensor.shape})` (equivalent to `[torch.Tensor.shape]({torch.Tensor.shape})`)

## Notes

- We hardcode a PyTorch version (e.g. to 2.11: `https://docs.pytorch.org/docs/2.11/`) instead of using `stable`. To update, change `PYTORCH_VERSION` in `PytorchLink.vue`.
- Certain functions that are available in PyTorch cannot be found in the PyTorch docs, such as `torch.Tensor.relu`.
