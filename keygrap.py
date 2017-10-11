import os
import time
import pickle

from PIL import Image
from pymouse import PyMouse
from pykeyboard import PyKeyboard
from mss import mss
import matplotlib

import pyxhook
from env import DarkSoulsEnv

m = PyMouse()
k = PyKeyboard()
sct = mss()

env = DarkSoulsEnv(sct=sct, m=m, k=k)

key_list = []

def OnKeyPress(event):
    global key_list
    key_list.append(event.Key)
    if event.Ascii==96:
        new_hook.cancel()

screen_path = '/home/leesy714/dataset/ds/'
dirs = os.listdir(screen_path)
numbers = [int(x[:-4]) for x in dirs]
if not numbers:
    max_number = -1
else:
    max_number = max(numbers)
new_hook=pyxhook.HookManager()
new_hook.KeyDown=OnKeyPress
new_hook.HookKeyboard()
new_hook.start()


target = dict()
if os.path.exists('dict.pkl'):
    target = pickle.load(open('dict.pkl','r'))

for i_episode in range(0,10):
    env.init_env()
    time.sleep(2.0)

    screen, hp, sp, boss_hp, estus_left, reward = env.status()
    restore = False
    boss_hp_zero_count = 0
    key_list=[]
    while True:
        im = Image.fromarray(screen.transpose((1,2,0)))

        time.sleep(2.0)
        if key_list:
            max_number += 1
            if 'h' in key_list:
                target[max_number]='light_atk'
            elif 'u' in key_list:
                target[max_number]='heavy_atk'
            elif 'e' in key_list:
                target[max_number]='drink_estus'

            elif 'space' in key_list:
                if 's' in key_list:
                    target[max_number]='b_roll'
                elif 'w' in key_list:
                    target[max_number]='f_roll'
                elif 'a' in key_list:
                    target[max_number]='l_roll'
                elif 'd' in key_list:
                    target[max_number]='r_roll'
            else:
                if 's' in key_list:
                    target[max_number]='move_backward'
                elif 'w' in key_list:
                    target[max_number]='move_forward'
                elif 'a' in key_list:
                    target[max_number]='move_left'
                elif 'd' in key_list:
                    target[max_number]='move_right'
                else:
                    target[max_number]='unknown'
            try:
                print target[max_number]
                im.save(screen_path+'%i.png'%max_number)
            except KeyError:
                max_number -= 1

        key_list=[]

        if hp < 0.0001 :
            break
        if boss_hp == 0:
            boss_hp_zero_count += 1
        else:
            boss_hp_zero_count = 0
        if boss_hp_zero_count > 10:
            restore = True
            break
        screen, hp, sp, boss_hp, estus_left, reward = env.status()


