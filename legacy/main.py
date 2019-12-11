
# coding: utf-8

# In[ ]:
import os
import argparse
import sys
import inspect
import time
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

from model import PolicyCNN

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
parser.add_argument('--gamma', default=0.99, type=float,
        help='Discount (default=0.99)')

SavedAction = namedtuple('SavedAction', ['action','distribution','value'])


mean = np.array( [ 0.19464481,  0.18079188,  0.14531441] )
mean = np.transpose(np.tile(mean,(450,800,1)),(2,0,1))
std = np.array([0.11782318,  0.11146703,  0.10342609])
std = np.transpose(np.tile(std,(450,800,1)),(2,0,1))




def draw_bar(label, action_data, value):
    with open('dist.log','w') as w:
        for l,d in zip(label,action_data):
            w.write('{} : {:.4f}\n'.format(l,d))
        w.write('value:{}\n'.format(value))

def forward_test(model, state, variables):
    state = torch.from_numpy(state).float().unsqueeze(0)
    variables = torch.from_numpy(variables).float().unsqueeze(0)
    probs, state_value = model(
                Variable(state, volatile=True).cuda(),
                Variable(variables, volatile=True).cuda()
            )



def select_action(model, state, variables):
    state = torch.from_numpy(state).float().unsqueeze(0)

    variables = torch.from_numpy(variables).float().unsqueeze(0)

    probs, state_value = model(
                Variable(state).cuda(),
                Variable(variables).cuda()
            )


    probs, state_value = probs.cpu(), state_value.cpu()
    pr = probs.data.numpy()[0]

    draw_bar(move_list, pr, state_value.data.numpy()[0,0]) 

    #action = probs.multinomial()
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(action, m, state_value))

    return action.data[0]


def finish_episode(model,optimizer):
    R = 0
    saved_actions = model.saved_actions
    rewards = []
    value_loss = 0
    loss = 0
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (action, m, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0,0]
        loss += -m.log_prob(action) * reward
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    optimizer.zero_grad()
    #final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    #gradients = [torch.ones(1)] + [None] * len(saved_actions)

    #autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
    #loss = loss + value_loss
    loss.backward(retain_graph=True)

    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]
    return R


def save_state(episode,model):
    path = os.path.join('runs',args.name)
    if not os.path.exists(path):
        os.makedirs(path)
    name = (episode/20) % 10

    torch.save(
            {'state_dict': model.state_dict(), 'episode':episode},
            path+'/%i_checkpoint.pth.tar'%(name))




def train(model, env, optimizer):

    screen_path = '/home/leesy714/dataset/ds/'
    dirs = os.listdir(screen_path)
    numbers = [int(x[:-4]) for x in dirs]
    if not numbers:
        max_number = -1
    else:
        max_number = max(numbers)
        
    time.sleep(2)

    n_move = len(move_list)
    
    i_step = 0
    for i_episode in range(args.start_episode + 1, args.episodes+1):

        screen, hp, sp, boss_hp, estus_left, reward = env.status()

        screen = (screen / 127.5)  - 1.0
        variables = np.array([hp - 0.5, boss_hp - 0.5, estus_left / 5.0 - 0.5])

        forward_test(model, screen, variables )
        env.init_env()
        time.sleep(2.0)
        start = time.time()
        gain = 0.0

        prev_action = np.random.randint(1,n_move)
        current_action = getattr(movement,move_list[prev_action])
        restore = False
        boss_hp_zero_count = 0
        screen, hp, sp, boss_hp, estus_left, reward = env.status()
        while True:
            prev = np.eye(n_move)[prev_action].flatten()

            if (boss_hp == 0 or hp ==0) and time.time()-start>5 and time.time()-start<10:
                restore = True
                break
            if boss_hp == 0:
                boss_hp_zero_count += 1
            else:
                boss_hp_zero_count = 0
            if boss_hp_zero_count > 10:
                restore = True
                break

            screen = (screen / 127.5)  - 1.0
            variables = np.array([hp - 0.5, boss_hp - 0.5, estus_left / 5.0 - 0.5])

            action = select_action(model, screen, variables)
            current_action = getattr(movement, move_list[action])
            prev_action=action
    
            screen, hp, sp, boss_hp, estus_left, reward = env.do_action(action=current_action)
            max_number += 1
            
            model.rewards.append(reward)
            gain += reward

            write ="{} {:.2f} {}  hp:{:.0f}  boss:{:.0f}  estus:{}  reward:{:.4f}  gain:{:.4f}".format(
                i_episode, time.time() - start, move_list[action], hp * 100.0, boss_hp * 100.0, estus_left, reward, gain)

            print(write)
            with open('temp_log.log','w') as w:
                w.write(write)
            write2 = "{},{},{},{},{},{},{},{}\n".format(
                i_episode, time.time() - start, move_list[action], hp * 100.0, boss_hp * 100.0, estus_left, reward, gain)



            if hp < 0.0001 and time.time()-start > 10:
                break
            if time.time()-start>180:
                restore = True
                break

            i_step += 1
        if restore:
            time.sleep(5)
            env.restore_save_file()
            print("Episode {} skipped. Restore save file.".format(i_episode))
            restore=False
            continue

        if i_episode % 20 == 0:
            save_state(i_episode, model)
        print()
        R=finish_episode(model=model, optimizer=optimizer)
        print("Episode {} R:{:.4f} gain:{:.4f}".format(i_episode,R,gain))
        with open('temp_log2.log','w') as w:
            w.write("Episode {} R:{:.4f} gain:{:.4f}".format(i_episode,R,gain))


        time.sleep(30)

    print("Trained {} episodes".format(args.episodes))






    
def main():
    global args, move_list
    args = parser.parse_args()


    #move_list = [x.__name__ for x in movement.__dict__.values()
    #    if inspect.isfunction(x)]
    #move_list.remove('focus')
    move_list=[]
    move_list.append('idle')
    move_list.append('f_roll')
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
    
    model = PolicyCNN(action=len(move_list), pretrained=args.pretrain, variables=3).cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    args.start_episode = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_episode = checkpoint['episode']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['episode']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train(model=model, env=env, optimizer=optimizer)
    
    




if __name__ == '__main__':
    main()
