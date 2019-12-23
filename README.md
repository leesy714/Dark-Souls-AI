# Dark-Souls-AI

AI for Dark Souls.

## Network Architecture
- 32 8x8 conv 4 stride
- 64 5x5 conv 2 stride
- 128 3x3 conv
- 256 3x3 conv
- Global average pooling
- concatenate hp, stamina, boss hp data(256->259)
- fully connected 259 -> 512
- fully connected 512 -> num action(default 7)
- softmax


## Arlgorithm
- RAINBOW <https://arxiv.org/abs/1710.02298>
- Pytorch codes from <https://github.com/belepi93/pytorch-rainbow>


## Configs
Environment optimized for Dark soul III Iudex  Gundyr

Screen is captured with mss <https://python-mss.readthedocs.io/index.html>
Keyboard input by PyUserInput <https://pypi.org/project/PyUserInput/>

Dark Souls III game must be running on full screen mode on a monitor.
Need to change some key maps for a proper run.

Proper save data is needed in data dir to restore initial position.

