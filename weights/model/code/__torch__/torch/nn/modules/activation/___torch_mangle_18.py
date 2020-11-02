class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_18.ReLU,
    argument_1: Tensor) -> Tensor:
    return torch.relu(argument_1)
