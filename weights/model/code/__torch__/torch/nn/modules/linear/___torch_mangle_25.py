class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_25.Linear,
    argument_1: Tensor) -> Tensor:
    _0 = torch.addmm(self.bias, argument_1, torch.t(self.weight), beta=1, alpha=1)
    return _0
