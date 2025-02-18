import torch
import torch.nn as nn

class KAN_Layer(nn.Module):
    def __init__(self, in_features, out_features, knots_dim, bsplines_generator):
      super(KAN_Layer, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.knots_dim = knots_dim
      self.knots = nn.Parameter(torch.randn(in_features, out_features, knots_dim))
      self.bsplines_generator = bsplines_generator

    def forward(self, x):
      b_of_x = torch.stack([self.bsplines_generator(x, i) for i in range(self.knots_dim)], dim=-1)
      repeat_dim = -2
      repeat_shape = [1] * (b_of_x.dim() + 1)
      repeat_shape[repeat_dim] = self.out_features
      splines = b_of_x.unsqueeze(repeat_dim).repeat(*repeat_shape)
      scalar = (self.knots * splines).sum(-1)
      y = scalar.sum(-2)
      return y