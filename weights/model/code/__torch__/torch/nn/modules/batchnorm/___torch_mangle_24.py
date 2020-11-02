class BatchNorm1d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_24.BatchNorm1d,
    argument_1: Tensor) -> Tensor:
    _0 = self.bias
    input = torch.batch_norm(argument_1, self.weight, _0, None, None, True, 0.10000000000000001, 1.0000000000000001e-05, True)
    return input
