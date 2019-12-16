import numpy as np
import back
import movement
import gym

from back import ConsoleInput
from screen import ScreenControl

class DarkSoulEnv(object):
    def __init__(self, action_set=None):
        self.s = ScreenControl()
        self.c = ConsoleInput()
        self.ready=False
        if action_set is None:
            self.action_set=[
                'light_atk','b_roll','f_roll','l_roll','r_roll','drink_estus','idle']
        else:
            self.action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self.action_set))
        self.obs_space = (1080,1920,3)



    def reset(self):
        back.exit_and_reload(self.c, focus=False)
        back.move_to_init_position(self.c,focus=False)
        self.ready=True
        self.estus = 3.0
        self.previous=(1.0, 1.0)
        img = np.array(self.s.get_screen())
        return img

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

        if hp < 0.001 or boss_hp < 0.001:
            done=True
        else:
            done=False
        reward = (hp - self.previous[0]) + (self.previous[1] - boss_hp)
        if np.abs(reward)<0.001:
            reward=0
        self.previous = (hp,boss_hp)

        return (img, hp, sp, boss_hp, self.estus), reward, done, None

class GrayscaleEnv(object):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.action_set = self.env.action_set
        self.obs_space = (1080,1920)

    def reset(self):
        img = self.env.reset()
        img = np.mean(img, axis=2)
        return img

    def step(self, action):
        (img, hp, sp, boss_hp, estus), reward, done, info = self.env.step(action)
        img = np.mean(img, axis=2)
        return (img, hp, sp, boss_hp, estus), reward, done, info

class CropPoolEnv(object):
    def __init__(self, env, crop=210, pool=4):
        self.env = env
        self.action_space = self.env.action_space
        self.action_set = self.env.action_set
        self.pool = pool
        self.crop = crop
        self.obs_space = (1080//pool,(1980-2*crop)//pool)

    def reset(self):
        img = self.env.reset()
        img = img[::self.pool,self.crop:-self.crop:self.pool]
        return img

    def step(self, action):
        (img, hp, sp, boss_hp, estus), reward, done, info = self.env.step(action)
        img = img[::self.pool,self.crop:-self.crop:self.pool]
        return (img, hp, sp, boss_hp, estus), reward, done, info



class FrameStackEnv(object):
    def __init__(self, env, frame=4):
        self.env = env
        self.action_space = self.env.action_space
        self.action_set = self.env.action_set
        self.obs_space = (frame,self.env.obs_space[0],self.env.obs_space[1])
        self.frame = frame
        self.img_list = []

    def reset(self):
        img = self.env.reset()
        self.img_list=[]
        self.img_list.append(img)
        idle = self.env.action_set.index('idle')
        for _ in range(1, self.frame):
            (img, hp, sp, boss_hp, estus), reward, done, info = self.env.step(idle)
            self.img_list.append(img)

        return np.array(self.img_list)

    def step(self, action):
        (img, hp, sp, boss_hp, estus), reward, done, info = self.env.step(action)
        self.img_list.append(img)
        self.img_list = self.img_list[1:]
        return (np.array(self.img_list), hp, sp, boss_hp, estus), reward, done, info



