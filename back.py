import time
from pykeyboard import PyKeyboard
from pymouse import PyMouse


class ConsoleInput():
    def __init__(self, pykey=None, pymouse=None):
        if pykey is None:
            pykey = PyKeyboard()
        if pymouse is None:
            pymouse = PyMouse()
        self.k = pykey
        self.m = pymouse
    def mouse_click(self,x,y):
        self.m.click(x,y,1)

    def press_sec(self, keys, sec=1, delay=0.1):
        for i, key in enumerate(keys):
            if i==0:
                self.k.press_key(key)
            else:
                self.k.press_key(key)
                time.sleep(0.1)
                self.k.release_key(key)

        time.sleep(sec)
        for i, key in enumerate(keys):
            self.k.release_key(key)
        time.sleep(delay)


if __name__ == '__main__':
    c = ConsoleInput()
    c.press_sec('K')
    c.mouse_click(960,520)

