import numpy as np
from mss import mss
from PIL import Image



class ScreenControl(object):
    def __init__(self):
        self.sct = mss()
    def get_screen(self, pos=None):
        if pos is None:
            pos = {'top':0,'left':0,'width':1920,'height':1080}
        sct_img = self.sct.grab(pos)
        img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
        img = img.convert('RGB')  # Convert to RGB

        return img

    def get_health(self):
        img = self.get_screen(pos={'top':77,'left':205,'width':251,'height':10})
        return img

    def get_stamina(self):
        img = self.get_screen(pos={'top':111,'left':200,'width':220,'height':10})
        return img

    def get_boss_health(self):
        img = self.get_screen(pos={'top':900,'left':560,'width':1000,'height':10})
        return img




    def get_hp_sp(self,
                    hp_path='./data/health.npy', sp_path='./data/stamina.npy'):

        ch,mh = np.array(self.get_health()),np.load(hp_path)
        cs,ms = np.array(self.get_stamina()), np.load(sp_path)
        hp = (np.sum((ch==mh).min(axis=2))/ch.shape[0]) / ch.shape[1]
        sp = (np.sum((cs==ms).min(axis=2))/cs.shape[0]) / cs.shape[1]
        return hp, sp

    def get_boss_hp(self, hp_path='./data/boss_hp.npy'):
        ch, mh = np.array(self.get_boss_health()), np.load(hp_path)
        hp = (np.sum((ch==mh).min(axis=2))/ch.shape[0]) / ch.shape[1]
        return hp










if __name__ == '__main__':
    s = ScreenControl()
    print(s.get_screen())
