import numpy as np
from mss import mss
from PIL import Image


def get_screen(sct):
    sct_img = sct.grab({'top':120,'left':102,'width':800,'height':450})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB

    return img

def get_soul(sct):
    sct_img = sct.grab({'top':520,'left':760,'width':80,'height':18})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB


    return screen(sct, mon={'top':567,'left':900,'width':100,'height':20})

def get_health(sct):
    sct_img = sct.grab({'top':169,'left':211,'width':320,'height':12})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB

    return img 

def get_stamina(sct):
    sct_img = sct.grab({'top':181,'left':211,'width':184,'height':12})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB

    return img 


def get_boss_health(sct):
    sct_img = sct.grab({'top':487,'left':348,'width':418,'height':12})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB

 

    return img


def get_estus_left(sct):
    sct_img = sct.grab({'height':12,'left':254,'top':523,'width':10})
    img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
    img = img.convert('RGB')  # Convert to RGB

    return img
