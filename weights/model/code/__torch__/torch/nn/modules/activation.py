class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    argument_1: Tensor) -> Tensor:
    return torch.relu(argument_1)
