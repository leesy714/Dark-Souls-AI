
# coding: utf-8

# In[ ]:
import copy
import math
import os
import argparse
import sys
import inspect
import time
import random
from collections import namedtuple
from itertools import count
from PIL import Image

from pymouse import PyMouse
from pykeyboard import PyKeyboard
from mss import mss

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import movement
import screen
from env import DarkSoulsEnv
from model import DQN

parser = argparse.ArgumentParser(description='Dark Souls Reinforcement Learning')
parser.add_argument('--episodes', default = 100, type=int,
        help='Number of total episodes (default=100)')
parser.add_argument('--name', default='model', type=str,
        help='Model name (default=model)')
parser.add_argument('--resume', default='', type=str,
        help='Load checkpoint and continue learning')
parser.add_argument('--pretrain', default='', type=str,
        help='Load pretrained feature model')
parser.add_argument('--lr', default=1e-4, type=float,
        help='Learning rate (default=1e-4)')
parser.add_argument('--batch-size', default=32, type=int,
        help='Batch size (default=32)')
parser.add_argument('--gamma', default=0.99, type=float,
        help='Discount (default=0.99)')
parser.add_argument('--alter-model', default=500, type=int,
        help='Alter model freq (default=500)')
parser.add_argument('--eps-start', default=0.9, type=float,
        help='Eps start (default=0.9)')
parser.add_argument('--eps-end', default=0.1, type=float,
        help='Eps end (default=0.1)')
parser.add_argument('--eps-decay', default=200, type=int,
        help='Eps decay freq(default=200)')

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



Transition = namedtuple('Transition', ('state','variables', 'action', 'next_state','next_variables', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory()

def draw_bar(label, action_value):
    with open('dist.log','w') as w:
        for l,d in zip(label,action_value):
            w.write('{} : {:.4f}\n'.format(l,d))
            #print('{} : {:.4f}'.format(l,d))

def select_action(model, state, variables):
    global i_step
    sample = random.random()

    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * i_step / args.eps_decay)
    i_step += 1

    if sample > eps_threshold:
        values = model(Variable(state, volatile=True).type(FloatTensor), Variable(variables, volatile=True).type(FloatTensor)).data
        draw_bar(move_list, values.cpu().numpy()[0])
        return values.max(1)[1].view(1,1)
    else:
        #print("Random walk by {:.3f} < {:.3f}".format(sample, eps_threshold))
        return LongTensor([[random.randrange(len(move_list))]])



def save_state(episode,model):
    global i_step
    path = os.path.join('runs',args.name)
    if not os.path.exists(path):
        os.makedirs(path)
    name = (episode/20) % 10

    torch.save(
            {'state_dict': model.state_dict(), 'episode':episode, 'step':i_step,
                'name':args.name},
            path+'/%i_checkpoint.pth.tar'%(name))
    torch.save(
            {'state_dict': model.state_dict(), 'episode':episode, 'step':i_step,
                'name':args.name},
            path+'/latest_checkpoint.pth.tar')


def optimize(model, target_model, optimizer ):
    global i_step
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    non_final_next_variables = Variable(torch.cat([s for s in batch.next_variables if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    variable_batch = Variable(torch.cat(batch.variables))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch, variable_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(args.batch_size).type(Tensor))
    next_state_values[non_final_mask] = target_model(non_final_next_states, non_final_next_variables).max(1)[0]

    next_state_values.volatile=False

    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    #loss = F.mse_loss(state_action_values, expected_state_action_values)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    print("Loss:{:.6f}".format(loss.cpu().data.numpy()[0]))

    with open('temp_log2.log','w') as w:
        w.write("Loss:{:.6f}\n".format(loss.cpu().data.numpy()[0]))
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1.,1.)
    optimizer.step()

def train(model,model2, env, optimizer):
    global i_step
    screen_path = '/home/leesy714/dataset/ds/'
    dirs = os.listdir(screen_path)
    numbers = [int(x[:-4]) for x in dirs]
    if not numbers:
        max_number = -1
    else:
        max_number = max(numbers)
        
    time.sleep(2)

    n_move = len(move_list)
    

    final_boss_hp=[]
    for i_episode in range(args.start_episode + 1, args.episodes+1):

        env.init_env()
        time.sleep(2.0)
        start = time.time()
        gain = 0.0

        restore = False
        boss_hp_zero_count = 0
        current_screen, hp, sp, boss_hp, estus_left, reward = env.status()
        current_screen = (current_screen / 127.5)  - 1.0
        current_screen = torch.from_numpy(current_screen).unsqueeze(0).type(Tensor)

        while True:
            if (boss_hp == 0 or hp ==0) and time.time()-start>5 and time.time()-start<10:
                restore = True
                break
            if boss_hp == 0:
                boss_hp_zero_count += 1
            else:
                boss_hp_zero_count = 0
            if boss_hp_zero_count > 5:
                restore = True
                break
            variables = np.array([hp - 0.5, boss_hp - 0.5, estus_left / 5.0 - 0.5])
            variables = torch.from_numpy(variables).unsqueeze(0).type(Tensor)
            action = select_action(model, current_screen, variables)
            current_action = getattr(movement, move_list[action[0,0]])
    
            screen, hp, sp, boss_hp, estus_left, reward = env.do_action(action=current_action)
            next_variables = np.array([hp - 0.5, boss_hp - 0.5, estus_left / 5.0 - 0.5])
            next_variables = torch.from_numpy(next_variables).unsqueeze(0).type(Tensor)

            screen = (screen / 127.5) - 1.0
            screen = torch.from_numpy(screen).unsqueeze(0).type(Tensor)

            gain += reward
            write ="{} {:.2f} {}  hp:{:.0f}  boss:{:.0f}  estus:{}  reward:{:.4f}  gain:{:.4f}".format(i_step, time.time() - start, move_list[action[0,0]], hp * 100.0, boss_hp * 100.0, estus_left, reward, gain)
            print(write)
            reward = Tensor([reward])
            if hp < 0.0001 and time.time()-start > 10:
                screen = None
                next_variables = None
            if boss_hp > 0.01 :
                memory.push(current_screen, variables, action, screen, next_variables, reward)
            else:
                print("Odd situation. State not saved.")
            
 
            with open('temp_log.log','w') as w:
                w.write(write)
            if boss_hp==-1.0 :
                print('Victory!')
                final_boss_hp.append(boss_hp)
                break

            if hp < 0.0001 and time.time()-start > 10:
                print('Dead')
                final_boss_hp.append(boss_hp)
                break
            if time.time()-start>180:
                restore = True
                break
            current_screen = screen 
            optimize(model, model2, optimizer)
            if i_step % args.alter_model == 0:
                alter_model(model, model2)
                print("Model shifted")

        if restore:
            time.sleep(5)
            env.restore_save_file()
            print("Episode {} skipped. Restore save file.".format(i_episode))
            restore=False
            continue

        if i_episode % 20 == 0:
            save_state(i_episode, model)
        print()
        print("Episode {} gain:{:.4f}".format(i_episode,gain))
        with open('temp_log2.log','w') as w:
            w.write("Episode {} gain:{:.4f}".format(i_episode,gain))
        if os.path.isfile('gain_log.log'):
            with open('gain_log.log','a') as w:
                w.write('{} {}\n'.format(i_episode, gain))
        else:
            with open('gain_log.log','w') as w:
                w.write('{} {}\n'.format(i_episode, gain))
 


        #experience replay for 30 sec
        #er_start = time.time()
        #while time.time() - er_start < 30.0:
        #    if len(memory) > args.batch_size:
        #        optimize(model, model2, optimizer)
        time.sleep(30)
                


    print("Trained {} episodes".format(args.episodes))





def alter_model(model1, model2):
    torch.save(
            {'state_dict': model1.state_dict()},
            'temp.pth.tar')
    model2 = copy.deepcopy(model1)

    checkpoint = torch.load('temp.pth.tar')
    model2.load_state_dict(checkpoint['state_dict'])

    #checkpoint = torch.load('temp2.pth.tar')
    #model1.load_state_dict(checkpoint['state_dict'])

    
def main():
    global args, move_list, i_step
    args = parser.parse_args()


    #move_list = [x.__name__ for x in movement.__dict__.values()
    #    if inspect.isfunction(x)]
    #move_list.remove('focus')
    move_list=[]
    move_list.append('f_roll')
    move_list.append('idle')
    move_list.append('r_roll')
    move_list.append('l_roll')
    move_list.append('b_roll')
    move_list.append('light_atk')
    move_list.append('drink_estus')




    m = PyMouse(display=':0')
    k = PyKeyboard(display=':0')
    sct = mss(display=':0')

    env = DarkSoulsEnv(sct=sct, m=m, k=k)




    if args.pretrain:
        pass
    else:
        args.pretrain=None
    
    model1 = DQN(action=len(move_list), variables=3, pretrained=args.pretrain)
    if use_cuda:
        model1.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model1.parameters()])))

    optimizer = optim.Adam(model1.parameters(), lr=args.lr)
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    i_step = 0
    args.start_episode = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_episode = checkpoint['episode']

            i_step = checkpoint['step']
            args.name = checkpoint['name']
            model1.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['episode']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model2 = copy.deepcopy(model1)



    train(model=model1, model2=model2, env=env, optimizer=optimizer)
    
    




if __name__ == '__main__':
    main()
