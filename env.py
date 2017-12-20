import os

import numpy as np


from screen import * 
import back
import movement
import time

def health_percent(img):
    max_y=0
    cnt = 0
    health_img = np.load('health_img.npy').astype(int)
    for y in range(0,320,5):
        point = img[4,y]
        if np.all(img[4,y]>=health_img[4,y]-10) and np.all(img[4,y]<=health_img[4,y]+10):
           max_y = y
            
    return max_y/320.0

def boss_health_percent(img):
    max_y=0
    cnt=0
    boss_img = np.load('boss_img.npy').astype(int)
    for y in range(0, 418, 5):
        point = img[4,y]
        if np.all(img[4,y] >= boss_img[4,y] - 10) and np.all(img[4,y] <= boss_img[4,y] + 10):
            max_y=y
        else:
            break


    return max_y/float(418.0)
 

def stamina_percent(img):
    max_y=0
    cnt = 0
    stamina_img = np.load('stamina_img.npy').astype(int)
    for y in range(0,184,5):
        point = img[4,y]
        if np.all(img[4,y] >= stamina_img[4,y]-10) and np.all(img[4,y] <= stamina_img[4,y]+10):
            max_y=y
        else:
            break
    return max_y/184.0

def get_boss_health_value(sct):
    img = np.asarray(get_boss_health(sct=sct))
    return boss_health_percent(img)

def get_health_value(sct):
    img = np.asarray(get_health(sct=sct))
    return health_percent(img)

def get_stamina_value(sct):
    img = np.asarray(get_stamina(sct=sct))
    return stamina_percent(img)

def get_estus_left_value(sct):
    img = np.asarray(get_estus_left(sct=sct))
    estus0=np.load('estus_0.npy')
    estus1=np.load('estus_1.npy')
    estus2=np.load('estus_2.npy')
    estus3=np.load('estus_3.npy')
    estus4=np.load('estus_4.npy')
    estus5=np.load('estus_5.npy')
    current_estus = (np.sum(img,axis=2)>400).astype(int)
    if np.sum((current_estus != estus5).astype(int))<3:
        return 5

    elif np.sum((current_estus != estus4).astype(int))<3:
        return 4

    elif np.sum((current_estus != estus3).astype(int))<3:
        return 3

    elif np.sum((current_estus != estus2).astype(int))<3:
        return 2

    elif np.sum((current_estus != estus1).astype(int))<3:
        return 1
    else:
        return 0



class DarkSoulsEnv():

    def __init__(self, sct, m, k):
        self.sct = sct
        self.m = m
        self.k = k
        self.prev_hp = 1.0
        self.prev_boss_hp = 1.0


    def init_env(self):
        back.focus_window(self.m)
        back.press_sec(self.k, keys=[self.k.control_l_key], sec=0.1)
        movement.focus(self.k)
        movement.move_forward(self.k, sec=4)
        back.press_sec(self.k, keys=['q'], sec=1)
        time.sleep(6)
        movement.move_forward(self.k, sec=3)
        time.sleep(2.0)
        movement.focus(self.k)
        self.prev_hp = 1.0
        self.prev_boss_hp = 1.0

    def status(self):
        screen = np.asarray(get_screen(sct=self.sct))

        estus_left = get_estus_left_value(sct=self.sct)

        #screen = screen / 255.0
        #screen = screen - 0.5
        screen = np.transpose(screen, axes=[2,0,1])

        hp = get_health_value(sct=self.sct) 
        sp = get_stamina_value(sct=self.sct)
        boss_hp = get_boss_health_value(sct=self.sct)
        if hp == 0 and boss_hp == 0:
            boss_hp = self.prev_boss_hp
        
        boss_hp_diff = self.prev_boss_hp - boss_hp
        hp_diff = self.prev_hp - hp

        reward = 0
        #if hp<0.001:
        #    reward = -1.0
        #elif boss_hp <0.001:
        #    reward = 1.0
        reward = boss_hp_diff * 10.0
        if hp<=0.01:
            reward = reward - 10.0
            if boss_hp <= 0.01:
                reward += 10.0
                boss_hp = -1.0
        if boss_hp<= 0.01 and boss_hp_diff < 0.11 and boss_hp_diff >0.01:
            boss_hp = -1.0
        
        return screen, hp, sp, boss_hp, estus_left, reward

   

    def do_action(self, action):
        action(k=self.k)
        screen, hp, sp, boss_hp, estus_left, reward = self.status()
        self.prev_boss_hp = boss_hp
        self.prev_hp = hp
        return screen, hp, sp, boss_hp, estus_left, reward

    
    def get_boss_hp(self):
        return get_boss_health_value(sct=self.sct)
    
    def get_hp(self):
        return get_health_value(sct=self.sct)

    def get_sp(self):
        return get_stamina_value(sct=self.sct )

    def restore_save_file(self):
        time.sleep(2)
        back.focus_window(self.m)
        time.sleep(1)
        back.press_sec(self.k, keys=[self.k.end_key],sec=0.2)
        time.sleep(1)
        for i in range(3):
            back.press_sec(self.k, keys=[self.k.right_key],sec=0.1)
            time.sleep(1)
        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.1)
        time.sleep(1)
        for i in range(4):
            back.press_sec(self.k, keys=[self.k.down_key],sec=0.1)
            time.sleep(1)

        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.1)
        time.sleep(1)

        back.press_sec(self.k, keys=[self.k.left_key],sec=0.1)
        time.sleep(1)
        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.1)

        time.sleep(5)
        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.2)
        time.sleep(1)
       
        cmd ='cp ~/.wine/drive_c/users/leesy714/My\ Documents/NBGI/DarkSouls/DRAKS0005_2.sl2 ~/.wine/drive_c/users/leesy714/My\ Documents/NBGI/DarkSouls/DRAKS0005.sl2'
        os.system(cmd)
        time.sleep(3)


        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.2)
        time.sleep(1)
        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.2)
        time.sleep(1)
        back.press_sec(self.k, keys=[self.k.enter_key],sec=0.2)
        time.sleep(10)



