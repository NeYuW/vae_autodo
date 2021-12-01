import torch
import kornia
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def hsv2rgb(input):
    assert(input.shape[1] == 3)

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb

def color(input, color_paras):
  output = (input ** color_paras[:, :3, None, None]) * (color_paras[:,3:6, None, None] + 1) + color_paras[:,6:, None, None]
  return output

def gkern(kernlen=5, nsig=1):

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return torch.Tensor(kern2d/kern2d.sum())

def ikern(kernlen=5):
  x = torch.zeros([kernlen, kernlen])
  x[int((kernlen - 1)/2), int((kernlen - 1)/2)] = 1
  return x

def filt(input, kernel_size, sig, s):
  N, C, H, W = input.shape
  g_kern = gkern(kernel_size, sig)
  i_kern = ikern(kernel_size)
  
  weights_list = []
  for _ in range(N):
    weight = nn.Parameter((s[_] * g_kern + i_kern).unsqueeze(0).unsqueeze(0).repeat(3,1,1,1))
    weights_list.append(weight)

  outputs = []
  for idx in range(N):
    inputs = input[idx:idx+1]
    filt = weights_list[idx]
    output = F.conv2d(inputs, filt, groups = 3, padding = 'same')
    outputs,append(output)

  outputs = torch.stack(outputs)
  outputs = outputs.squeeze(1)
  return outputs

def geo(input, geo_paras):
  N, C, H, W = input.shape
  geo_matrix = geo_paras.view([-1, 3, 3])
  outputs = kornia.geometry.transform.warp_perspective(input, geo_matrix, (H, W))

  return outputs
  

