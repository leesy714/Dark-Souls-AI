def move_forward(c, sec=0.5):
    c.press_sec(keys=['w'], sec=sec)

def move_backward(c, sec=0.5):
    c.press_sec(keys=['s'], sec=sec)

def move_left(c, sec=0.5):
    c.press_sec(keys=['a'], sec=sec)

def move_right(c, sec=0.5):
    c.press_sec(keys=['d'], sec=sec)

def run_backward(c, sec=0.5):
    c.press_sec(keys=[' ','s'], sec=sec)

def focus(c):
    c.press_sec(keys=['q'], sec=0.1)

def idle_(c):
    c.press_sec(keys=['x'], sec=0.0, delay=0.5)

def light_atk(c,delay=0.2):
    c.press_sec(keys=['u'], sec=0.3, delay=delay)

def heavy_atk(c,delay=0.2):
    c.press_sec(keys=['u'], sec=0.3, delay=delay)

def f_roll(c, sec=0.1):
    c.press_sec(keys=['w',' '], sec=sec, delay=0.4)

def b_roll(c, sec=0.1):
    c.press_sec(keys=['s',' '], sec=sec, delay=0.4)

def l_roll(c, sec=0.1):
    c.press_sec(keys=['a',' '], sec=sec, delay=0.4)

def r_roll(c, sec=0.1):
    c.press_sec(keys=['d',' '], sec=sec, delay=0.4)

def drink_estus(c, sec=0.1):
    c.press_sec(keys=['r'], sec=sec,delay=2.4)

if __name__ == '__main__':
    from back import ConsoleInput
    c = ConsoleInput()
    move_forward(c)





