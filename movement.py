import back

def move_forward(k, sec=1.0):
    back.press_sec(k, keys=['w'], sec=sec)

def move_backward(k, sec=1.0):
    back.press_sec(k, keys=['s'], sec=sec)

def move_left(k, sec=1.0):
    back.press_sec(k, keys=['a'], sec=sec)

def move_right(k, sec=1.0):
    back.press_sec(k, keys=['d'], sec=sec)

def focus(k):
    back.press_sec(k, keys=['o'], sec=0.2)

def idle(k):
    back.press_sec(k, keys=['x'], sec=0.0, delay=0.3)

def light_atk(k):
    back.press_sec(k, keys=['h'], sec=0.2, delay=0.5)

def heavy_atk(k):
    back.press_sec(k, keys=['u'], sec=0.2, delay=1.8)

def f_roll(k, sec=0.2):
    back.press_sec(k, keys=['w',' '], sec=1.0)

def b_roll(k, sec=0.2):
    back.press_sec(k, keys=['s',' '], sec=1.0)

def l_roll(k, sec=0.2):
    back.press_sec(k, keys=['a',' '], sec=1.0)

def r_roll(k, sec=0.2):
    back.press_sec(k, keys=['d',' '], sec=1.0)

def drink_estus(k, sec=0.2):
    back.press_sec(k, keys=['e'], sec=2.7)





