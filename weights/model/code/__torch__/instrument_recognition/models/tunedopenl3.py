class TunedOpenL3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  filters : __torch__.instrument_recognition.models.timefreq.Melspectrogram
  openl3 : __torch__.instrument_recognition.models.torchopenl3.OpenL3Mel128
  fc_seq : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.instrument_recognition.models.tunedopenl3.TunedOpenL3,
    input: Tensor) -> Tensor:
    _0 = self.fc_seq
    _1 = (self.openl3).forward((self.filters).forward(input, ), )
    return (_0).forward(_1, )
