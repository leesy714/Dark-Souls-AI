
# coding: utf-8

# In[ ]:
import os
import argparse
import sys
import time
import inspect
from itertools import count
import random
import math 
from collections import namedtuple

from pymouse import PyMouse
from pykeyboard import PyKeyboard
from mss import mss

from PIL import Image

import matplotlib
from matplotlib import pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import movement
import screen
from env import DarkSoulsEnv
from model import DQN

parser = argparse.ArgumentParser(description='Dark Souls Reinforcement Learning')
parser.add_argument('--episodes', default = 100, type=int,
        help='Number of total episodes (default=100)')
parser.add_argument('--resume', default='', type=str,
        help='Load checkpoint and continue learning')
parser.add_argument('--lr', default=1e-5, type=float,
        help='Learning rate (default=1e-5)')
parser.add_argument('--batch-size', default=32, type=int,
        help='Batch size (default=32)')
parser.add_argument('--gamma', default=0.999, type=float,
        help='Discount (default=0.999)')
parser.add_argument('--eps-start', default=0.9, type=float,
        help='Epsilon start (default=0.9)')
parser.add_argument('--eps-end', default=0.3, type=float,
        help='Epsilon start (default=0.3)')
parser.add_argument('--eps-decay', default=200, type=int,
        help='Epsilon decay frequency (default=200)')

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
        from IPython import display
plt.ion()
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def process_screen(screen):
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).type(Tensor)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, alpha=0.5):
        self.reset(alpha=0.5)
        
    def reset(self, alpha):
        self.val = 0
        self.alpha = alpha

    def update(self, val):
        self.val = self.val * (1 - self.alpha) + val * self.alpha



def select_action(state):
    global steps_done,num_action
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay)
    steps_done += 1

    action_values = model(Variable(state, volatile=True).type(FloatTensor)).data
    if sample > eps_threshold:
        print(action_values.cpu().numpy()[0], move_list[ action_values.max(1)[1].view(1,1)[0,0]])
        return action_values.max(1)[1].view(1,1)
    else:
        act = random.randrange(num_action)
        print('Random', move_list[act])
        return LongTensor([[act]])




def save_state(episode,model):
    path = os.path.join('runs','test')
    if not os.path.exists(path):
        os.makedirs(path)
    name = episode%10

    torch.save(
            {'state_dict': model.state_dict(), 'episode':episode},
            path+'/%i_checkpoint.pth.tar'%(name))


def optimize_model():
    global last_sync
    if len(memory) < args.batch_size:
        return

    transitions = memory.sample(args.batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
        batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),
            volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(args.batch_size).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    #for param in model.parameters():
    #    param.grad.data.clamp_(-1,1)
    optimizer.step()



args = parser.parse_args()
move_list = [x.__name__ for x in movement.__dict__.values() if inspect.isfunction(x)]
move_list.remove('focus')
m = PyMouse()
k = PyKeyboard()
sct = mss()

env = DarkSoulsEnv(sct=sct, m=m, k=k)
num_action = len(move_list)

 
model = DQN(action=num_action)

memory = ReplayMemory(10000)
steps_done = 0
last_sync = 0


criterion = nn.MSELoss()
if use_cuda:
    model.cuda()
    criterion = criterion.cuda()
#optimizer = optim.RMSprop(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=args.lr)
memory = ReplayMemory(10000)


# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

start_episode = 0
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_episode = checkpoint['episode']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['episode']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

num_episodes = args.episodes
for i_episode in range(start_episode, num_episodes):
    env.init_env()
    last_screen, hp, sp, boss_hp, estus_left, reward = env.status()
    last_screen = process_screen(last_screen)
    gain = 0.0
    time.sleep(2.0)

    current_screen, hp, sp, boss_hp, estus_left, reward = env.status()
    current_screen = process_screen(current_screen)
    
    state = current_screen - last_screen
    start = time.time()

    boss_hp_zero_count = 0

    restore = False
    temp_memory = [] 
    gain = 0

    for t in count():
        action = select_action(state)
        current_action = getattr(movement, move_list[action[0,0]])

        last_screen = current_screen
        current_screen, hp, sp, boss_hp, estus_left, reward = env.do_action(action=current_action)
        gain += reward
        current_screen = process_screen(current_screen)
        reward = Tensor([reward])

        
        if hp < 0.0001 and time.time()-start > 10:
            break
        if time.time()-start>300:
            restore = True
            break

        next_state = current_screen - last_screen
        temp_memory.append((state,action,next_state,reward))
        state = next_state
    print('Episode ',i_episode,'Done\tGain:',gain )

    if restore:
        time.sleep(5)
        env.restore_save_file()
        print("Episode {} skipped. Restore save file.".format(i_episode))
        restore=False
        continue
    else:
        for state, action, next_state, reward in temp_memory:
            memory.push(state,action,next_state, reward)
        for op in range(len(temp_memory)):
            optimize_model()
    save_state(i_episode, model)
   


    
    time.sleep(50)
print("Trained {} episodes".format(num_episodes))







