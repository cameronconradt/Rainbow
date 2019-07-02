import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

# os.makedirs("images", exist_ok=True)
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
# opt = parser.parse_args()
# print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)




class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class WganDiv:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.steps = 0
        self.k = 2
        self.p = 6
        self.lr = .0002
        self.b1 = .5
        self.b2 = .999
        self.latent_dim = 100
        self.n_critic = 5
        print(int(np.prod(img_shape)))

        # Initialize generator and discriminator
        self.generator = Generator(img_shape, self.latent_dim)
        self.discriminator = Discriminator(img_shape)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.batches_done = 0

    def train_step(self, imgs, next_img):
        # Configure input
        next_img = Variable(next_img.type(self.Tensor), requires_grad=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

        # Generate an image
        fake_img = self.generator(z)

        # Real image
        real_validity = self.discriminator(next_img)
        # Fake image
        fake_validity = self.discriminator(fake_img)

        # Compute W-div gradient penalty
        real_grad_out = Variable(self.Tensor(next_img.size(0), 1).fill_(1.0), requires_grad=False)
        real_grad = autograd.grad(
            real_validity, next_img, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)

        fake_grad_out = Variable(self.Tensor(fake_img.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            fake_validity, fake_img, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * self.k / 2

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

        d_loss.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if self.steps % self.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_img = self.generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = self.discriminator(fake_img)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            self.optimizer_G.step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )

            # if self.batches_done % opt.sample_interval == 0:
            #     save_image(fake_img.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            #
            # self.batches_done += opt.n_critic

        self.steps += 1

    def gen_image(self, imgs):
        z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))
        return self.generator(z)






# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

