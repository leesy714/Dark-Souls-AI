import numpy as np
from mss import mss
from PIL import Image



class ScreenControl(object):
    def __init__(self):
        self.sct = mss()
        self.shape = (1080,1920,3)
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

class GrayscaleScreenControl(object):
    def __init__(self, ctrl=None):
        if ctrl is None:
            self.ctrl = ScreenControl()
        else:
            self.ctrl = ctrl
        self.shape = (1080,1920)
    def get_screen(self, pos=None):
        img = self.ctrl.get_screen(pos=pos)
        img = np.mean(img, axis=2)/255
        return img
    def get_health(self):
        return self.ctrl.get_health()
    def get_stamina(self):
        return self.ctrl.get_stamina()
    def get_boss_health(self):
        return self.ctrl.get_boss_health()

    def get_hp_sp(self):
        return self.ctrl.get_hp_sp()

    def get_boss_hp(self):
        return self.ctrl.get_boss_hp()

class CropPoolScreenControl(object):
    def __init__(self, ctrl=None, crop=420, pool=8):
        if ctrl is None:
            self.ctrl = GrayscaleScreenControl()
        else:
            self.ctrl = ctrl
        self.crop=crop
        self.pool=pool
        if len(self.ctrl.shape)==3:
            self.shape = (
                self.ctrl.shape[0],
                self.ctrl.shape[1] // pool,
                (self.ctrl.shape[2] - 2*crop) // pool)
        else:
            self.shape = (
                self.ctrl.shape[0] // pool,
                (self.ctrl.shape[1] - 2*crop) // pool)

    def get_screen(self, pos=None):
        img = self.ctrl.get_screen(pos=pos)
        if len(self.shape)==3:
            img = img[:,::self.pool,self.crop:-self.crop:self.pool]
        else:
            img = img[::self.pool,self.crop:-self.crop:self.pool]
        return img

    def get_health(self):
        return self.ctrl.get_health()
    def get_stamina(self):
        return self.ctrl.get_stamina()
    def get_boss_health(self):
        return self.ctrl.get_boss_health()

    def get_hp_sp(self):
        return self.ctrl.get_hp_sp()

    def get_boss_hp(self):
        return self.ctrl.get_boss_hp()

class AsyncFrameStackScreenControl(object):
    def __init__(self, ctrl=None, frame=4, fps=4.0):
        if ctrl is None:
            self.ctrl = CropPoolScreenControl()
        else:
            self.ctrl = ctrl
        print(self.ctrl.shape)
        assert self.ctrl.shape!=2
        self.frame = frame
        self.fps = fps
        self.spf = 1.0//fps
        self.shape = (frame, self.ctrl.shape[0], self.ctrl.shape[1])
        self.img_list = []
        self.capturer = None

    def start(self):
        import threading
        import time
        self.lock = threading.Lock()

        def capture(ctrl, frame, spf):
            while True:
                with self.lock:
                    img = ctrl.get_screen()
                    self.img_list.append(img)
                    if len(self.img_list)>frame:
                        self.img_list = self.img_list[-frame:]
                time.sleep(spf)

        self.capturer = threading.Thread(
            target = capture,
            args = (self.ctrl, self.frame, self.spf))
        self.capturer.start()

    def get_screen(self, pos=None):
        with self.lock:
           return np.array(self.img_list[-self.frame:])


    def get_health(self):
        with self.lock:
            return self.ctrl.get_health()
    def get_stamina(self):

        with self.lock:
            return self.ctrl.get_stamina()
    def get_boss_health(self):
        with self.lock:
            return self.ctrl.get_boss_health()

    def get_hp_sp(self):
        with self.lock:
            return self.ctrl.get_hp_sp()

    def get_boss_hp(self):
        with self.lock:
            return self.ctrl.get_boss_hp()
















if __name__ == '__main__':
    s = ScreenControl()
    print(s.get_screen())
