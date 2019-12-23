import numpy as np
import time
import back
import movement
import gym

from back import ConsoleInput
from screen import ScreenControl,AsyncFrameStackScreenControl

class DarkSoulEnv(object):
    def __init__(self, action_set=None):
        self.s = AsyncFrameStackScreenControl()
        self.c = ConsoleInput()
        self.ready=False
        if action_set is None:
            self.action_set=[
                #'light_atk','b_roll','f_roll','l_roll','r_roll','move_forward','move_backward','move_left','move_right','drink_estus','idle_']
                'light_atk','b_roll','f_roll','l_roll','r_roll','drink_estus','idle_']
        else:
            self.action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self.action_set))
        self.observation_space = gym.spaces.Box(0,1.0,self.s.shape)
        self.s.start()



    def reset(self):
        back.exit_and_reload(self.c, focus=False)
        back.move_to_init_position(self.c,focus=False)
        self.ready=True
        self.estus = 3.0
        self.previous=(1.0, 1.0)
        img = np.array(self.s.get_screen())

        info = {'hp':1.0, 'boss_hp':1.0, 'sp':1.0,'estus':3.0}
        return img, info

    def step(self, action):
        if self.action_set[action] == 'drink_estus':
            pass
            #self.estus -= 1.0
            #if self.estus<0:
            #    self.estus=0.0
        getattr(movement, self.action_set[action])(self.c)
        img = np.array(self.s.get_screen())
        hp,sp = self.s.get_hp_sp()
        boss_hp = self.s.get_boss_hp()

        if hp < 0.0001 or boss_hp < 0.0001:
            done=True
        else:
            done=False
        #reward = (hp - self.previous[0]) + (self.previous[1] - boss_hp)
        reward=0.0
        #if hp-self.previous[0]>0.01:
        #    reward = 0.1
        #if self.previous[0]-hp>0.01:
        #    reward = -0.1
        if hp<0.0001:
            reward -= 1.0
        #if np.abs(self.previous[0] - hp)>0.01:
        #    reward = hp - self.previous[0]
        if np.abs(self.previous[1] - boss_hp)>0.01:
            reward += 0.1
        if np.abs(reward)<0.001:
            reward=0
        self.previous = (hp,boss_hp)
        info = {'hp':hp, 'boss_hp':boss_hp, 'sp':sp,'estus':self.estus}

        #return (img, hp, sp, boss_hp, self.estus), reward 10, done, None
        return img, reward * 10, done, info

