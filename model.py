import torch
import kornia
import torch.nn as nn

def bright(input, bright_paras, device):
    """brightness adjustment function"""
    bright_paras = (torch.sigmoid(bright_paras) * 0.4).to(device)   # range:[0, 0.4]
    outputs = kornia.enhance.adjust_brightness(input, bright_paras).to(device)
    return outputs

def contrast(input, contrast_paras, device):
    """contrast adjustment function"""
    contrast_paras = (torch.sigmoid(contrast_paras) * 0.5 + 0.5).to(device)   # range:[0.5, 1]
    outputs = kornia.enhance.adjust_contrast(input, contrast_paras).to(device)
    return outputs

def saturation(input, saturation_paras, device):
    """saturation adjustment function"""
    saturation_paras = (torch.sigmoid(saturation_paras) * 2).to(device)   # range:[0, 2]
    outputs = kornia.enhance.adjust_saturation(input, saturation_paras).to(device)
    return outputs

def sharp(input, sharp_paras, device):
    """sharpness adjustment function"""
    sharp_paras = (torch.sigmoid(sharp_paras) * 5).to(device)    # range:[0, 5]
    outputs = kornia.enhance.sharpness(input, sharp_paras).to(device)
    return outputs

def affine(input, affine_paras, device):
    """affine transformation function"""
    B, C, H, W = input.shape
    affine_paras = (torch.sigmoid(affine_paras) * 0.5 + 1).to(device)   # range:[1, 1.5]
    helper = torch.eye(2,3).repeat(B, 1, 1).to(device)
    paras = helper * affine_paras.unsqueeze(2).to(device)
    print(paras)

def rotate(input, rotate_paras, device):
    """rotation transformation function"""
    rotate_paras = (torch.sigmoid(rotate_paras) * 60).to(device)   # range:[0, 60]
    outputs = kornia.geometry.transform.rotate(input, rotate_paras).to(device)
    return outputs

def translate(input, translate_paras, device):
    """translation transformation function"""
    translate_paras = (torch.sigmoid(translate_paras) * 10 - 5).to(device)   # range:[-5, 5]
    outputs = kornia.geometry.transform.translate(input, translate_paras).to(device)
    return outputs

def scale(input, scale_paras, device):
    """scale adjustment function"""
    scale_paras = (torch.sigmoid(scale_paras) * 1.5 + 0.5).to(device)   # range:[0.5, 2]
    outputs = kornia.geometry.transform.scale(input, scale_paras).to(device)
    return outputs

def shear(input, shear_paras, device):
    """shear adjustment function"""
    shear_paras = (torch.sigmoid(shear_paras) * 0.6 - 0.3).to(device)   # range:[-0.3, 0.3]
    outputs = kornia.geometry.transform.shear(input, shear_paras).to(device)
    return outputs


class VAE(nn.Module):
    """use variational autoencoder as the strategy network"""
    def __init__(self, image_size=32, in_channels=3, h_dim=None, z_dim=20):
        super(VAE, self).__init__()
        self.z_dim = z_dim   # dimension of latent variables
        self.size = image_size  # image size

        # Build Encoder
        modules = list()
        if h_dim is None:
            h_dim = [32, 64, 128, 256]
        for dim in h_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels=dim,
                        kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU())
            )
            in_channels = dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(4 * h_dim[-1], z_dim)
        self.fc_var = nn.Linear(4 * h_dim[-1], z_dim)

        # Build Decoder
        modules = list()
        self.decoder_input = nn.Linear(z_dim, h_dim[-1])
        h_dim.reverse()
        for i in range(len(h_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim[i], h_dim[i + 1]),
                    nn.BatchNorm1d(h_dim[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(h_dim[-1], 11)

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_var(h)  # two encoders for mu and var, respectively

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = self.decoder(h)
        h = self.final_layer(h)
        return h

    def forward(self, x, device):
        mu, log_var = self.encode(x)
        mu = mu.to(device)
        log_var = log_var.to(device)
        z = self.reparameterize(mu, log_var)
        hp = self.decode(z)
        # Data augmentation starts
        x = bright(x, hp[:, 0], device)
        x = contrast(x, hp[:, 1], device)
        x = saturation(x, hp[:, 2], device)
        x = sharp(x, hp[:, 3], device)
        x = scale(x, hp[:, 4:6], device)
        x = rotate(x, hp[:, 6], device)
        x = translate(x, hp[:, 7:9], device)
        x_reconstruct = shear(x, hp[:, 9:], device)
        # Data augmentation ends
        x = scale(x, hp[:, 4:6], device)
        x = rotate(x, hp[:, 6], device)
        x = translate(x, hp[:, 7:9], device)
        x = shear(x, hp[:, 9:], device)
        x = bright(x, hp[:, 0], device)
        x = contrast(x, hp[:, 1], device)
        x = saturation(x, hp[:, 2], device)
        x_reconstruct = sharp(x, hp[:, 3], device)

        return x_reconstruct, mu, log_var

    def loss(self, mu, logvar, device):
        """kdloss"""
        kdloss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0).to(device)
        return kdloss
