class Melspectrogram(Module):
  __parameters__ = ["mel_filters", ]
  __buffers__ = []
  mel_filters : Tensor
  training : bool
  conv1d_real : __torch__.torch.nn.modules.conv.Conv1d
  conv1d_imag : __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv1d
  freq2mel : __torch__.torch.nn.modules.linear.Linear
  def forward(self: __torch__.instrument_recognition.models.timefreq.Melspectrogram,
    input: Tensor) -> Tensor:
    _0 = self.mel_filters
    _1 = self.conv1d_imag
    _2 = (self.conv1d_real).forward(input, )
    _3 = (_1).forward(input, )
    x = torch.add(torch.pow(_2, 2), torch.pow(_3, 2), alpha=1)
    x0 = torch.permute(x, [0, 2, 1])
    x1 = torch.matmul(x0, torch.numpy_T(_0))
    x2 = torch.permute(x1, [0, 2, 1])
    x3 = torch.pow(torch.sqrt(x2), 1.)
    x4 = torch.view(x3, [-1, 1, 128, 199])
    _4 = torch.full_like(x4, 1e-10, dtype=6, layout=0, device=torch.device("cpu"), pin_memory=False, memory_format=None)
    amin = torch.to(_4, 6, False, False, None)
    _5 = torch.to(CONSTANTS.c0, torch.device("cpu"), 7, False, False, None)
    log10 = torch.to(torch.detach(_5), 6, False, False, None)
    _6 = torch.mul(torch.log(torch.max(x4, amin)), CONSTANTS.c1)
    x5 = torch.div(_6, log10)
    xmax, _7 = torch.max(x5, 1, True)
    xmax0, _8 = torch.max(xmax, 2, True)
    xmax1, _9 = torch.max(xmax0, 3, True)
    x6 = torch.sub(x5, xmax1, alpha=1)
    return torch.clamp(x6, -80., None)
