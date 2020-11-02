class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.___torch_mangle_7.BatchNorm2d,
    argument_1: Tensor) -> Tensor:
    _0 = self.running_var
    _1 = self.running_mean
    _2 = self.bias
    input = torch.batch_norm(argument_1, self.weight, _2, _1, _0, False, 0., 0.0010000000474974513, True)
    return input
