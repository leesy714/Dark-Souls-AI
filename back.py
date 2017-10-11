import time

x_dim, y_dim = 1920, 1080

def focus_window(m):
    m.click(int(x_dim/4), int(y_dim/4), 1)
def press_sec(k, keys, sec=1, delay=0.1):
    for i, key in enumerate(keys):
        if i==0:
            k.press_key(key)
        else:
            k.press_key(key)
            time.sleep(0.1)
            k.release_key(key)

    time.sleep(sec)
    for i, key in enumerate(keys):
        k.release_key(key)
    time.sleep(delay)
