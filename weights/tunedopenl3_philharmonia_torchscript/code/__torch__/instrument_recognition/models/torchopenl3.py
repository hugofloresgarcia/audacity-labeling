class OpenL3Mel128(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  batch_normalization_1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  conv2d_1 : __torch__.torch.nn.modules.conv.Conv2d
  batch_normalization_2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_1.BatchNorm2d
  conv2d_2 : __torch__.torch.nn.modules.conv.___torch_mangle_2.Conv2d
  batch_normalization_3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_3.BatchNorm2d
  conv2d_3 : __torch__.torch.nn.modules.conv.___torch_mangle_4.Conv2d
  batch_normalization_4 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_5.BatchNorm2d
  conv2d_4 : __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d
  batch_normalization_5 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_7.BatchNorm2d
  conv2d_5 : __torch__.torch.nn.modules.conv.___torch_mangle_8.Conv2d
  batch_normalization_6 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_9.BatchNorm2d
  conv2d_6 : __torch__.torch.nn.modules.conv.___torch_mangle_10.Conv2d
  batch_normalization_7 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_11.BatchNorm2d
  conv2d_7 : __torch__.torch.nn.modules.conv.___torch_mangle_12.Conv2d
  batch_normalization_8 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_13.BatchNorm2d
  audio_embedding_layer : __torch__.torch.nn.modules.conv.___torch_mangle_14.Conv2d
  maxpool : __torch__.torch.nn.modules.pooling.MaxPool2d
  def forward(self: __torch__.instrument_recognition.models.torchopenl3.OpenL3Mel128,
    argument_1: Tensor) -> Tensor:
    _0 = self.maxpool
    _1 = self.audio_embedding_layer
    _2 = self.batch_normalization_8
    _3 = self.conv2d_7
    _4 = self.batch_normalization_7
    _5 = self.conv2d_6
    _6 = self.batch_normalization_6
    _7 = self.conv2d_5
    _8 = self.batch_normalization_5
    _9 = self.conv2d_4
    _10 = self.batch_normalization_4
    _11 = self.conv2d_3
    _12 = self.batch_normalization_3
    _13 = self.conv2d_2
    _14 = self.batch_normalization_2
    _15 = self.conv2d_1
    _16 = (self.batch_normalization_1).forward(argument_1, )
    input = torch.constant_pad_nd(_16, [1, 1, 1, 1], 0)
    _17 = (_14).forward((_15).forward(input, ), )
    input0 = torch.relu(_17)
    input1 = torch.constant_pad_nd(input0, [1, 1, 1, 1], 0)
    _18 = (_12).forward((_13).forward(input1, ), )
    input2 = torch.relu(_18)
    input3, _19 = torch.max_pool2d_with_indices(input2, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input4 = torch.constant_pad_nd(input3, [1, 1, 1, 1], 0)
    _20 = (_10).forward((_11).forward(input4, ), )
    input5 = torch.relu(_20)
    input6 = torch.constant_pad_nd(input5, [1, 1, 1, 1], 0)
    _21 = (_8).forward((_9).forward(input6, ), )
    input7 = torch.relu(_21)
    input8, _22 = torch.max_pool2d_with_indices(input7, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input9 = torch.constant_pad_nd(input8, [1, 1, 1, 1], 0)
    _23 = (_6).forward((_7).forward(input9, ), )
    input10 = torch.relu(_23)
    input11 = torch.constant_pad_nd(input10, [1, 1, 1, 1], 0)
    _24 = (_4).forward((_5).forward(input11, ), )
    input12 = torch.relu(_24)
    input13, _25 = torch.max_pool2d_with_indices(input12, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input14 = torch.constant_pad_nd(input13, [1, 1, 1, 1], 0)
    _26 = (_2).forward((_3).forward(input14, ), )
    input15 = torch.relu(_26)
    input16 = torch.constant_pad_nd(input15, [1, 1, 1, 1], 0)
    _27 = (_0).forward((_1).forward(input16, ), )
    return _27
