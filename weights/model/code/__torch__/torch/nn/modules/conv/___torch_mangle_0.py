class Conv1d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv1d,
    input: Tensor) -> Tensor:
    imag = torch._convolution(input, self.weight, None, [242], [1101], [1], False, [0], 1, False, False, True)
    return imag
