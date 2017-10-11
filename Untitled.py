
# coding: utf-8

# In[1]:

import numpy as np
import cv2
from mss import mss
from PIL import Image
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard


# In[2]:

sct = mss()


# In[16]:

def get_screen(sct, mon={'top':140, 'left':100, 'width':950, 'height':580}):
    pixel=sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    return img


# In[45]:

def get_soul(sct):
    return get_screen(sct, mon={'top':653,'left':920,'width':90,'height':20})


# In[57]:

def press_sec(m, k, keys, sec=1):
    m.click(x_dim/2, y_dim/2, 1)
    for key in keys:
        k.press_key(key)
    time.sleep(sec)
    for key in keys:
        k.release_key(key)
    


# In[85]:



# In[81]:


m = PyMouse()
k = PyKeyboard()


# In[82]:

x_dim, y_dim = m.screen_size()
x_dim, y_dim


# In[83]:

# F,B,L,R
# FJ, BJ, LJ, RJ
# R1R2 L1



# In[86]:

f_roll(m,k)


# In[ ]:



