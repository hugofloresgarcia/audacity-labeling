class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.max_pool2d(argument_1, [4, 8], [4, 8], [0, 0], [1, 1], False)
    return input
