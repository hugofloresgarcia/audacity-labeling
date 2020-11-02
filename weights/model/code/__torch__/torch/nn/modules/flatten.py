class Flatten(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.flatten.Flatten,
    argument_1: Tensor) -> Tensor:
    return torch.flatten(argument_1, 1, -1)
