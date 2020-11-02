class Linear(Module):
  __parameters__ = ["weight", "weights", ]
  __buffers__ = []
  weight : Tensor
  weights : Tensor
  training : bool
