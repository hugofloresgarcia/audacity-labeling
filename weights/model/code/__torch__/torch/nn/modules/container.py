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
  __annotations__["7"] = __torch__.torch.nn.modules.activation.___torch_mangle_18.ReLU
  __annotations__["8"] = __torch__.torch.nn.modules.dropout.___torch_mangle_19.Dropout
  __annotations__["9"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_20.BatchNorm1d
  __annotations__["10"] = __torch__.torch.nn.modules.linear.___torch_mangle_21.Linear
  __annotations__["11"] = __torch__.torch.nn.modules.activation.___torch_mangle_22.ReLU
  __annotations__["12"] = __torch__.torch.nn.modules.dropout.___torch_mangle_23.Dropout
  __annotations__["13"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_24.BatchNorm1d
  __annotations__["14"] = __torch__.torch.nn.modules.linear.___torch_mangle_25.Linear
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "14")
    _1 = getattr(self, "13")
    _2 = getattr(self, "12")
    _3 = getattr(self, "11")
    _4 = getattr(self, "10")
    _5 = getattr(self, "9")
    _6 = getattr(self, "8")
    _7 = getattr(self, "7")
    _8 = getattr(self, "6")
    _9 = getattr(self, "5")
    _10 = getattr(self, "4")
    _11 = getattr(self, "3")
    _12 = getattr(self, "2")
    _13 = getattr(self, "1")
    _14 = (getattr(self, "0")).forward(argument_1, )
    _15 = (_12).forward((_13).forward(_14, ), )
    _16 = (_10).forward((_11).forward(_15, ), )
    _17 = (_7).forward((_8).forward((_9).forward(_16, ), ), )
    _18 = (_4).forward((_5).forward((_6).forward(_17, ), ), )
    _19 = (_1).forward((_2).forward((_3).forward(_18, ), ), )
    return (_0).forward(_19, )
