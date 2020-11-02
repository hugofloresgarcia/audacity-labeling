class BatchNorm1d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm1d,
    argument_1: Tensor) -> Tensor:
    _0 = self.running_var
    _1 = self.running_mean
    _2 = self.bias
    input = torch.batch_norm(argument_1, self.weight, _2, _1, _0, False, 0.10000000000000001, 1.0000000000000001e-05, True)
    return input
