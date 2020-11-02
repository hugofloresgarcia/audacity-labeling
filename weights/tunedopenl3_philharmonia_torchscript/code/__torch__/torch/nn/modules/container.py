class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.flatten.Flatten
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.BatchNorm1d
  __annotations__["2"] = __torch__.torch.nn.modules.linear.___torch_mangle_15.Linear
  __annotations__["3"] = __torch__.torch.nn.modules.activation.ReLU
  __annotations__["4"] = __torch__.torch.nn.modules.dropout.Dropout
  __annotations__["5"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_16.BatchNorm1d
  __annotations__["6"] = __torch__.torch.nn.modules.linear.___torch_mangle_17.Linear
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "6")
    _1 = getattr(self, "5")
    _2 = getattr(self, "4")
    _3 = getattr(self, "3")
    _4 = getattr(self, "2")
    _5 = getattr(self, "1")
    _6 = (getattr(self, "0")).forward(argument_1, )
    _7 = (_3).forward((_4).forward((_5).forward(_6, ), ), )
    _8 = (_0).forward((_1).forward((_2).forward(_7, ), ), )
    return _8
