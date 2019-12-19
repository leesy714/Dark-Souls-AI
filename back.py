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

def restore_save_file():
    import os
    cmd = 'cp -f ./data/DS30000.sl2 /home/leesy714/.local/share/Steam/steamapps/compatdata/374320/pfx/drive_c/users/steamuser/Application\ Data/DarkSoulsIII/0110000102de41c4/'
    os.system(cmd)

def exit_and_reload(c, focus=False):
    #focus
    if focus:
        c.mouse_click(960,540)
        time.sleep(0.5)
    #exit
    c.press_sec(['q'],0.2)
    time.sleep(0.6)
    c.press_sec([c.k.escape_key],0.3)

    c.press_sec([c.k.right_key],0.5)

    c.press_sec(['e'],0.2)

    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)
    c.press_sec([ c.k.shift_l_key,c.k.right_key],0.2)

    c.press_sec(['e'],0.2)
    c.press_sec([c.k.left_key],0.1)
    c.press_sec(['e'],0.2)
    #wait for loading
    time.sleep(15)
    #restore save file
    restore_save_file()
    #restart game
    c.press_sec(['e'],0.2)
    time.sleep(2)
    c.press_sec(['e'],0.3)
    c.press_sec(['e'],0.3)
    time.sleep(0.3)
    c.press_sec(['e'],0.3)
    c.press_sec(['e'],0.3)
    time.sleep(5)
    return

def move_to_init_position(c, focus=False):
    if focus:
        c.mouse_click(960,540)
        time.sleep(0.1)
    c.press_sec('q',0.2)
    c.press_sec('w',3.0)
    c.press_sec('e',0.3)
    time.sleep(3)
    c.press_sec('w',4.5)
    c.press_sec('q',0.1)
    return



if __name__ == '__main__':
    c = ConsoleInput()
    c.press_sec('K')
    c.mouse_click(960,520)

