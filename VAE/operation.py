import torch
import kornia
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



def rgb2hsv(input, epsilon, device):
  return kornia.color.rgb_to_hsv(input, epsilon=1e-08).to(deivce)

def hsv2rgb(input, device):
  return kornia.color.hsv_to_rgb(input).to(device)

def color(input, color_paras, device):
  output = (input ** color_paras[:, :3, None, None]) * (color_paras[:,3:6, None, None] + 1) + color_paras[:,6:, None, None].to(device)
  output = nn.Sigmoid(output)
  helper = torch.ones(output.shape).to(device)
  helper[:, 0] = 2 * math.pi * helper[:, 0]
  outputs = (output * helper).to(device)

  return outputs

def geo(input, geo_paras, evice):
  N, C, H, W = input.shape
  geo_matrix = geo_paras.view([-1, 3, 3]).to(device)
  outputs = kornia.geometry.transform.warp_perspective(input, geo_matrix, (H, W)).to(device)

  return outputs
  

