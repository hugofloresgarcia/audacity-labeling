class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.BatchNorm2d,
    argument_1: Tensor) -> Tensor:
    _0 = self.bias
    input = torch.batch_norm(argument_1, self.weight, _0, None, None, True, 0., 0.0010000000474974513, True)
    return input
class BatchNorm1d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.BatchNorm1d,
    argument_1: Tensor) -> Tensor:
    _1 = self.bias
    input = torch.batch_norm(argument_1, self.weight, _1, None, None, True, 0.10000000000000001, 1.0000000000000001e-05, True)
    return input
