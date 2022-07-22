
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, noise_size = 128, input_size = 500):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.input_size = input_size
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(noise_size, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, input_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img




class Discriminator(nn.Module):
    def __init__(self, input_size = 500):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.model(img)
        out = self.final(out)
        return out
    
    def inter_loss(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        loss = torch.mean((x1-x2)**2)
        return loss
    
    def inter_loss_mse(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        loss = torch.mean((x1-x2)**2)
        return loss


class GAN_AE(torch.nn.Module):
    def __init__(self, noise_size=128, cuda = False):
        super().__init__()
        self.noise_size = noise_size
        self.generator = Generator(noise_size = noise_size)
        self.discriminator = Discriminator()
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), lr = 1e-4)
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr = 1e-4)
        self.loss = torch.nn.BCELoss()
        self.residual_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.7
        self.cuda = cuda

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss.cuda()
            self.residual_loss.cuda()

    def forward(self, state):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        fake_state = self.generator(noise)
        #print(fake_state.shape)
        #Train Generator
        self.optim_gen.zero_grad()
        gen_loss = self.loss(self.discriminator(fake_state), ones)
        gen_loss.backward()
        self.optim_gen.step()
        #Train Discriminator
        self.optim_disc.zero_grad()
        class_state = self.discriminator(state)
        disc_loss = (self.loss(class_state, ones) + \
            self.loss(self.discriminator(fake_state.detach()), zeros))/2
        disc_loss.backward()
        self.optim_disc.step()
        return fake_state.detach()
     
    def get_similarity_reward(self, state):
        state = state.unsqueeze(1)
        noise = torch.randn(1, self.noise_size)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
        noise.requires_grad_(True)
        optim_latent = torch.optim.Adam([noise], lr=5e-2)
        for i in range(10):
            optim_latent.zero_grad()
            fake_state = self.generator(noise)
            loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
            loss_v.backward()
            #print(loss_v.item())
            #print(fake_state)
            optim_latent.step()
        noise.detach()    
        fake_state = self.generator(noise)
        #print(fake_state)
        #loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
        loss_v = self.residual_loss(state, fake_state)
        return loss_v.item()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))
        x = np.float32(x) * (1.0 / 255.0)



class GAN(torch.nn.Module):
    def __init__(self, noise_size, generator, discriminator, lr_gen, lr_disc, cuda = False):
        super().__init__()
        self.noise_size = noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), lr = lr_gen)
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr = lr_disc)
        self.loss = torch.nn.BCELoss()
        self.residual_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = 0.7
        self.cuda = cuda

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.loss.cuda()
            self.residual_loss.cuda()

    def forward(self, state):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size, 1, 1)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        fake_state = self.generator(noise)
        #print(fake_state.shape)
        #Train Generator
        self.optim_gen.zero_grad()
        gen_loss = self.loss(self.discriminator(fake_state), ones)
        gen_loss.backward()
        self.optim_gen.step()
        #Train Discriminator
        self.optim_disc.zero_grad()
        class_state = self.discriminator(state)
        disc_loss = (self.loss(class_state, ones) + \
            self.loss(self.discriminator(fake_state.detach()), zeros))/2
        disc_loss.backward()
        self.optim_disc.step()
        return fake_state.detach()
     
    def get_similarity_reward(self, state):
        noise = torch.randn(1, self.noise_size, 1, 1)
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
        noise.requires_grad_(True)
        optim_latent = torch.optim.Adam([noise], lr=5e-2)
        for i in range(10):
            optim_latent.zero_grad()
            fake_state = self.generator(noise)
            loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
            loss_v.backward()
            #print(loss_v.item())
            #print(fake_state)
            optim_latent.step()
        noise.detach()    
        fake_state = self.generator(noise)
        #print(fake_state)
        #loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
        loss_v = self.residual_loss(state, fake_state)
        return loss_v.item()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))
        x = np.float32(x) * (1.0 / 255.0)

    def train(self, data, ep_num = 100, batchsize = 64):
        print("Training GAN")
        loader = DataLoader(data, batch_size=64, shuffle = True)
        for i in range(ep_num):
            print("Episode {}/{}".format(i, ep_num), end = '\r')
            for batch in loader:
                self.forward(batch)




class GAN_F(torch.nn.Module):
    def __init__(self, noise_size, generator, discriminator, encoder, lr_gen, lr_disc, lr_enc, cuda = False):
        super().__init__()
        self.noise_size = noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), lr = lr_gen, betas = (0.5, 0.999))
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr = lr_disc, betas = (0.5, 0.999))
        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr = lr_enc)
        self.loss = torch.nn.BCELoss()
        self.residual_loss = torch.nn.MSELoss(reduction='mean')
        self.weight_factor = 1
        self.lam = 0.7
        self.cuda = cuda

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.encoder.cuda()
            self.loss.cuda()
            self.residual_loss.cuda()

    def forward(self, state):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size, 1, 1)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        fake_state = self.generator(noise)
        #print(fake_state.shape)
        #Train Generator
        self.optim_gen.zero_grad()
        gen_loss = self.loss(self.discriminator(fake_state), ones)
        gen_loss.backward()
        self.optim_gen.step()
        #Train Discriminator
        self.optim_disc.zero_grad()
        class_state = self.discriminator(state)
        disc_loss = (self.loss(class_state, ones) + \
            self.loss(self.discriminator(fake_state.detach()), zeros))/2
        disc_loss.backward()
        self.optim_disc.step()
        #Train Encoder
        self.optim_enc.zero_grad()
        enc_z = self.encoder(state)
        dec_z = self.generator(enc_z)
        enc_loss = self.residual_loss(state, dec_z) +  self.discriminator.inter_loss(state, dec_z)
        enc_loss.backward()
        self.optim_enc.step()

        return enc_loss.item()
    
    def forward_gan(self, state):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size, 1, 1)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        fake_state = self.generator(noise)
        #print(fake_state.shape)
        #Train Generator
        self.optim_gen.zero_grad()
        gen_loss = self.loss(self.discriminator(fake_state), ones)
        gen_loss.backward()
        self.optim_gen.step()
        #Train Discriminator
        self.optim_disc.zero_grad()
        class_state = self.discriminator(state)
        disc_loss = (self.loss(class_state, ones) + \
            self.loss(self.discriminator(fake_state.detach()), zeros))/2
        disc_loss.backward()
        self.optim_disc.step()

    def forward_enc(self, state):
        state = state.unsqueeze(1)
        if self.cuda:
            state = state.cuda()
        self.optim_enc.zero_grad()

        enc_z = self.encoder(state)
        dec_z = self.generator(enc_z)
        enc_loss = self.residual_loss(state, dec_z) +  self.discriminator.inter_loss(state, dec_z)
        enc_loss.backward()
        
        self.optim_enc.step()

    
    def train(self, data, ep_num = 50 , batchsize = 64):
        print("Training GAN")
        loader = DataLoader(data, batch_size=64, shuffle = True)
        #train gan first
        for i in range(ep_num):
            print("Episode_GAN {}/{}".format(i, ep_num), end = '\r')
            for batch in loader:
                self.forward_gan(batch)
        #train enc second
        for i in range(ep_num):
            print("Episode_ENC {}/{}".format(i, ep_num), end = '\r')
            for batch in loader:
                self.forward_enc(batch)        
     
    def get_similarity_reward(self, state):
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if self.cuda:
            state = state.cuda()
        dec_z = self.generator(self.encoder(state))
        #print(dec_z)
        enc_loss = self.residual_loss(state, dec_z) +  self.discriminator.inter_loss(state, dec_z)
        return enc_loss.item()

    def get_similarity_reward_old(self, state):
        noise = torch.randn(1, self.noise_size, 1, 1)
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
        noise.requires_grad_(True)
        optim_latent = torch.optim.Adam([noise], lr=5e-2)
        for i in range(10):
            optim_latent.zero_grad()
            fake_state = self.generator(noise)
            loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
            loss_v.backward()
            #print(loss_v.item())
            #print(fake_state)
            optim_latent.step()
        noise.detach()    
        fake_state = self.generator(noise)
        #print(fake_state)
        #loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
        loss_v = self.residual_loss(state, fake_state)
        return loss_v.item()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

def compute_gradient_penalty(D, real_samples, fake_samples, cuda = True):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1))).float().cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class WGAN_F(nn.Module):
    def __init__(self, noise_size, generator, discriminator, encoder, lr_gen, lr_disc, lr_enc, k=1, cuda = False, n_critic = 5, lambda_gp = 10):
        super().__init__()
        self.noise_size = noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.optim_gen = torch.optim.RMSprop(self.generator.parameters(), lr = lr_gen)
        self.optim_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr = lr_disc)
        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr = lr_enc, betas = (0.0, 0.9))
        self.residual_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = k
        self.lambda_gp = lambda_gp
        self.cuda = cuda
        self.n_critic = n_critic

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.encoder.cuda()
            self.residual_loss.cuda()

    def forward(self, state):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size, 1, 1)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        fake_state = self.generator(noise)
        #print(fake_state.shape)
        #Train Generator
        self.optim_gen.zero_grad()
        gen_loss = self.loss(self.discriminator(fake_state), ones)
        gen_loss.backward()
        self.optim_gen.step()
        #Train Discriminator
        self.optim_disc.zero_grad()
        class_state = self.discriminator(state)
        disc_loss = (self.loss(class_state, ones) + \
            self.loss(self.discriminator(fake_state.detach()), zeros))/2
        disc_loss.backward()
        self.optim_disc.step()
        #Train Encoder
        self.optim_enc.zero_grad()
        enc_z = self.encoder(state)
        dec_z = self.generator(enc_z)
        enc_loss = self.residual_loss(state, dec_z) +  self.discriminator.inter_loss(state, dec_z)
        enc_loss.backward()
        self.optim_enc.step()

        return enc_loss.item()
    
    def forward_gan(self, state, train_gen = False):
        batch = state.shape[0]
        state = state.unsqueeze(1)
        noise = torch.randn(batch, self.noise_size, 1, 1)
        ones = torch.ones(batch, dtype=torch.float)
        zeros = torch.zeros(batch, dtype=torch.float)
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()
        #Train Discriminator
        self.optim_disc.zero_grad()
        fake_state = self.generator(noise)
        real_val = self.discriminator(state)
        fake_val = self.discriminator(fake_state.detach())
        #gradient_penalty = compute_gradient_penalty(self.discriminator, state.detach().clone(), fake_state.detach().clone())
        disc_loss = -torch.mean(real_val) + torch.mean(fake_val) #+ self.lambda_gp * gradient_penalty
        disc_loss.backward()
        self.optim_disc.step()
        #Clip weights of disc
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        #Train Generator
        if train_gen:
            self.optim_gen.zero_grad()
            #fake_state = self.generator(noise)
            fake_val = self.discriminator(fake_state)
            gen_loss = -torch.mean(fake_val)
            gen_loss.backward()
            self.optim_gen.step()

    def forward_enc(self, state):
        state = state.unsqueeze(1)
        if self.cuda:
            state = state.cuda()
        self.optim_enc.zero_grad()

        enc_z = self.encoder(state)
        dec_z = self.generator(enc_z)
        enc_loss = self.residual_loss(state, dec_z)  +  self.lam * self.discriminator.inter_loss(state, dec_z)
        enc_loss.backward()
        
        self.optim_enc.step()

    
    def train(self, data, ep_num = 50 , batchsize = 64):
        print("Training GAN")
        loader = DataLoader(data, batch_size=64, shuffle = True)
        print(len(loader))
        #train gan first
        for i in range(ep_num):
            print("Episode_GAN {}/{}".format(i, ep_num), end = '\r')
            for j, batch in enumerate(loader):
                is_train_gen = ( j % self.n_critic == 0)
                self.forward_gan(batch, train_gen = is_train_gen)
        #train enc second
        for i in range(ep_num):
            print("Episode_ENC {}/{}".format(i, ep_num), end = '\r')
            for batch in loader:
                self.forward_enc(batch)        
     
    def get_similarity_reward(self, state, save_img = False):
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if self.cuda:
            state = state.cuda()
        dec_z = self.generator(self.encoder(state))
        if save_img:
            img = dec_z.cpu().detach().numpy()
            img = np.squeeze(img)
            plt.imsave("images/new_{}.png", img, cmap="gray")

        #print(dec_z)
        
        enc_loss =  self.residual_loss(state, dec_z) #+ self.discriminator.inter_loss(state, dec_z)
        return enc_loss.item()

    def get_similarity_reward_old(self, state, save_img = False):
        noise = torch.randn(1, self.noise_size, 1, 1)
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if self.cuda:
            noise = noise.cuda()
            state = state.cuda()
        noise.requires_grad_(True)
        optim_latent = torch.optim.Adam([noise], lr=5e-2)
        for i in range(10):
            optim_latent.zero_grad()
            fake_state = self.generator(noise)
            loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
            loss_v.backward()
            #print(loss_v.item())
            #print(fake_state)
            optim_latent.step()
        noise.detach()    
        fake_state = self.generator(noise)
        if save_img:
            img = fake_state.cpu().detach().numpy()
            img = np.squeeze(img)
            plt.imsave("images/old.png", img, cmap="gray")

        #print(fake_state)
        #loss_v = (1-self.lam)*self.discriminator.inter_loss(state, fake_state) + self.lam*self.residual_loss(state, fake_state)
        loss_v = self.residual_loss(state, fake_state)
        return loss_v.item()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

