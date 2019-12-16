def move_forward(c, sec=1.0):
    c.press_sec(keys=['w'], sec=sec)

def move_backward(c, sec=1.0):
    c.press_sec(keys=['s'], sec=sec)

def move_left(c, sec=1.0):
    c.press_sec(keys=['a'], sec=sec)

def move_right(c, sec=1.0):
    c.press_sec(keys=['d'], sec=sec)

def run_backward(c, sec=1.0):
    c.press_sec(keys=[' ','s'], sec=sec)

def focus(c):
    c.press_sec(keys=['q'], sec=0.1)

def idle(c):
    c.press_sec(keys=['x'], sec=0.0, delay=0.1)

def light_atk(c,delay=0.5):
    c.press_sec(keys=['u'], sec=0.1, delay=delay)

def heavy_atk(c,delay=0.5):
    c.press_sec(keys=['u'], sec=0.1, delay=delay)

def f_roll(c, sec=0.2):
    c.press_sec(keys=['w',' '], sec=sec)

def b_roll(c, sec=0.2):
    c.press_sec(keys=['s',' '], sec=sec)

def l_roll(c, sec=0.2):
    c.press_sec(keys=['a',' '], sec=sec)

def r_roll(c, sec=0.2):
    c.press_sec(keys=['d',' '], sec=sec)

def drink_estus(c, sec=0.2):
    c.press_sec(keys=['r'], sec=0.1,delay=3.0)

if __name__ == '__main__':
    from back import ConsoleInput
    c = ConsoleInput()
    move_forward(c)





