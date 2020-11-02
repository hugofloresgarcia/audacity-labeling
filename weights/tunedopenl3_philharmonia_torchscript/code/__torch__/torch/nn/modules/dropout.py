class Dropout(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.dropout.Dropout,
    argument_1: Tensor) -> Tensor:
    input = torch.dropout(argument_1, 0.10000000000000001, False)
    return input
