import gym
import os
import random
from itertools import chain
from random import sample as randsample

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
import time
import datetime
from copy import deepcopy
import pickle

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

from tensorboardX import SummaryWriter
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace as BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from GAN.convolutional_ganmodels import Generator, Discriminator, Encoder
from GAN.gan_module import GAN, GAN_F, WGAN_F

def t(x): return torch.from_numpy(x).float()

class MovingAvgStd():
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.avg = 0
        self.var = 0
        self.size = 0
    def update(self, data):
        if self.size == 0:
            self.avg = data
        else:
            self.avg = self.alpha*data + (1-self.alpha)*self.avg
            self.var = self.alpha*((data - self.avg)**2) + (1-self.alpha)*self.var
        self.size += 1
        if self.var == 0:
            new_data = data
        else:
            new_data = (data - self.avg) / np.sqrt(self.var)
        
        return float(new_data)

class ReplayBuffer():
    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    @property
    def size(self):
        return len(self.queue)

    def __len__(self):
        return self.capacity

    def push(self, state):
        if (self.size == self.capacity):
            self.queue.pop()
        self.queue.insert(0, state)
    
    def sample(self, batchsize):
        assert self.size >= batchsize, "Buffer is not large enough!"
        sample = randsample(self.queue, batchsize)
        return sample

    def sample_end(self, batchsize):
        assert self.size >= batchsize, "Buffer is not large enough!"
        sample = self.queue[0:batchsize]
        return sample

    def clear(self):
        self.queue = []

class MontezumaEnv(Process):
    def __init__(self, render, env_idx, child_conn, history_size = 4, h = 84, w = 84):
        super(MontezumaEnv, self).__init__()
        self.env = gym.make("MontezumaRevengeNoFrameskip-v4")
        self.env_idx = env_idx
        self.render = render
        self.child_conn = child_conn
        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.init_state()

    def init_state(self):
        state = self.env.reset()
        state, _, _, _ = self.env.step(1)
        state = self.pre_proc(state)
        for i in range(self.history_size):
            self.history[i, :, :] = state.copy()

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (84, 84))
        x = np.float32(x) * (1.0 / 255.0)
        return x

    def run(self):
        super(MontezumaEnv, self).run()

        episode = 0
        steps = 0
        score = 0
        max_score = 0
        life = 5
        dead = False

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step(action)
            
            if life > info['ale.lives']:
                dead = True
                life = info['ale.lives']

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(next_state)
            
            steps += 1
            score += reward

            self.child_conn.send([deepcopy(self.history), reward, dead, done])

            if done and dead:
                if score > max_score:
                    max_score = score
                print('{} episode ({}) | score: {:2f} | max_score: {:2f} | steps: {}'.format(
                    episode, self.env_idx, score, max_score, steps
                ))
                episode += 1
                steps = 0
                score = 0
                dead = False
                life = 5
                self.init_state()

            if dead:
                dead = False
                self.init_state()

class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=True):
        self.model = CnnActorCriticNetwork(
            input_size, output_size, use_noisy_net)
        
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
       
        self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)
        

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(policy)

        return action

    def get_action_deterministic(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = np.argmax(policy)

        return action

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = eta * \
            ((real_next_state_feature - pred_next_state_feature).pow(2)).sum(1) / 2.
        return intrinsic_reward.data.cpu().numpy()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value = agent.model(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = agent.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def train_model(
            self,
            s_batch,
            next_s_batch,
            target_batch,
            y_batch,
            adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        if use_standardization:
            adv_batch = (adv_batch - adv_batch.mean()) / \
                (adv_batch.std() + stable_eps)

        ce = nn.CrossEntropyLoss()
        # mse = nn.SmoothL1Loss()
        forward_mse = nn.MSELoss()

        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * adv_batch

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        mse = nn.MSELoss()
        critic_loss = mse(value.sum(1), target_batch)

        self.optimizer.zero_grad()

        # Total loss
   
        loss = actor_loss.mean() + 0.5 * critic_loss - entropy_coef * entropy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        self.optimizer.step()


def make_train_data(reward, done, value, next_value):
    discounted_return = np.empty([num_step])

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[-1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t] + gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

        # For Actor
        adv = discounted_return - value

    return discounted_return, adv


if __name__ == '__main__':
    env = gym.make("MontezumaRevengeNoFrameskip-v4")
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    env.close()

    writer = SummaryWriter()
    use_cuda = True
    use_gae = True
    use_gan = True
    sparse_rew = True
    life_done = True

    is_load_model = False
    is_training = True

    is_render = False
    use_standardization = True
    use_noisy_net = True

    model_path = 'models/{}_{}.model'.format('MontezumaRevenge',
                                             datetime.date.today().isoformat())
    #load_model_path = 'models/MontezumaRevenge-v2_2018-09-18.model'

    lam = 0.95
    num_worker = 16
    num_step = 5
    max_step = 1.15e8


    learning_rate = 0.00025
    lr_schedule = False

    stable_eps = 1e-30
    entropy_coef = 0.02
    intr_rew_coef = 10
    alpha = 0.99
    gamma = 0.99
    clip_grad_norm = 0.5

    # Curiosity param
    lamb = 0.1
    beta = 0.2
    eta = 0.01
    batchsize = 64
    capacity_buffer = 2e4
    emw = MovingAvgStd(0.02)
    memory = ReplayBuffer(capacity_buffer)

    #Setup GAN
    noise_size = 128
    generator = Generator(noise_size)
    discriminator = Discriminator(None)
    encoder = Encoder(noise_size)
    lr_gen  = 1e-4
    lr_disc = 1e-4
    lr_enc = 1e-4
    gan_module = WGAN_F(noise_size, generator, discriminator, encoder, lr_gen, lr_disc, lr_enc, k = 1, n_critic=1, cuda=True)

    agent = ActorAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net)

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'))

    if not is_training:
        agent.model.eval()

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = MontezumaEnv(is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    visited = np.zeros((256, 4000))
    num_updates = 0
    mode = 0 # Changes to 1 after training GAN the first time. 0 = Don't add intrinsic reward, 1 = Add intrinsic reward
    furthest = 0
    furthest_limit = 0
    eval_rewards = []
    eval_steps = []
    #For evaluation 
    env_eval = gym.make("MontezumaRevengeNoFrameskip-v4")

    def proc(X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (84, 84))
        x = np.float32(x) * (1.0 / 255.0)
        return x

    def evaluate_agent(env, agent):
        total_reward = 0
        steps = 0
        states = np.zeros([4, 84, 84])
        init_s = proc(env.reset())
        for i in range(4):
            states[i, :, :] = init_s.copy()
        while True:
            action = agent.get_action_deterministic(np.expand_dims(states, axis=0))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            states[:3, :, :] = states[1:, :, :]
            states[3, :, :] = proc(obs)
            steps += 1
            if done:
                return (total_reward, steps)
    while True:
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            if not is_training:
                time.sleep(0.05)
            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones= [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd = parent_conn.recv()
                next_states.append(s)
                memory.push(t(s[3,:,:]))
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)  #rewards or log_rewards
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            if use_gan:
                if memory.size == memory.capacity:
                    data = torch.stack(memory.queue)
                    if mode == 1:
                        gan_module.train(data, ep_num = 40)
                    else:
                        gan_module.train(data, ep_num = 160)
                    memory.clear()
                    memory.capacity = 1e4
                    mode = 1
                    emw = MovingAvgStd(0.02)

                #Add GAN intrinsic_reward and train it
                if mode == 1:
                    gan_module.generator.eval()
                    gan_module.discriminator.eval()
                    last_states = t(next_states[:, 3, :, :]) #Only take the state agent is in, discard the history
                    for i in range(num_worker):
                        #gan_module.generator.eval()
                        intr_reward = gan_module.get_similarity_reward(last_states[i])
                        intr_reward = intr_rew_coef * emw.update(intr_reward)
                        rewards[i] += intr_reward
                    gan_module.generator.train()
                    gan_module.discriminator.train()
                #gan_module.forward(last_states)

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :, :, :]

            sample_rall += rewards[sample_env_idx]
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0

        if is_training:
            total_state = np.stack(total_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.stack(total_next_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_reward = np.stack(total_reward).transpose().reshape([-1])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose().reshape([-1])

            value, next_value, policy = agent.forward_transition(
                total_state, total_next_state)

            # logging output to see how convergent it is.
            policy = policy.detach()
            m = F.softmax(policy, dim=-1)
            recent_prob.append(m.max(1)[0].mean().cpu().numpy())
            writer.add_scalar(
                'data/max_prob',
                np.mean(recent_prob),
                sample_episode)

            total_target = []
            total_adv = []
            for idx in range(num_worker):
                target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                              total_done[idx *
                                                         num_step:(idx + 1) * num_step],
                                              value[idx *
                                                    num_step:(idx + 1) * num_step],
                                              next_value[idx * num_step:(idx + 1) * num_step])
                total_target.append(target)
                total_adv.append(adv)

            agent.train_model(
                total_state,
                total_next_state,
                np.hstack(total_target),
                total_action,
                np.hstack(total_adv))

            # last_states = total_state[:, 3, :, :]
            # gan_module.forward(t(last_states))

            # adjust learning rate
            if lr_schedule:
                new_learing_rate = learning_rate - \
                    (global_step / max_step) * learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_learing_rate
                    writer.add_scalar(
                        'data/lr', new_learing_rate, sample_episode)
            # Save the agent model periodically
            if global_step % (num_worker * num_step * 1000) == 0:
                torch.save(agent.model.state_dict(), model_path)

            # Save the rewards for plotting and evaluating purposes
            if global_step % (num_worker * num_step * 1000) == 0:
                agent.model.eval()
                rew, ev_steps= evaluate_agent(env_eval, agent)
                eval_rewards.append(rew)
                eval_steps.append(ev_steps)
                with open('plots/montezuma_gan_rewards', 'wb') as f:
                    pickle.dump(eval_rewards, f)
                with open('plots/montezuma_gan_steps', 'wb') as f:
                    pickle.dump(eval_steps, f)    
                agent.model.train()
            
            # Save the GAN model periodically
            if global_step % (num_worker * num_step * 5000) == 0:
                torch.save(gan_module.state_dict(), "models/GAN_{}.pth".format(global_step//(num_worker * num_step)))
