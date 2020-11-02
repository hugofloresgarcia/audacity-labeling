class Conv1d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.Conv1d,
    input: Tensor) -> Tensor:
    real = torch._convolution(input, self.weight, None, [242], [1101], [1], False, [0], 1, False, False, True)
    return real
class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    input: Tensor) -> Tensor:
    _0 = self.bias
    input0 = torch._convolution(input, self.weight, _0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)
    return input0
