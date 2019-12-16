from env import GrayscaleEnv,DarkSoulEnv, FrameStackEnv, CropPoolEnv
import numpy as np
import time
from tensorboardX import SummaryWriter
import pickle
from datetime import datetime

writer=SummaryWriter(logdir='./logs/instance_{}'.format(datetime.now()))


env = FrameStackEnv(CropPoolEnv(GrayscaleEnv(DarkSoulEnv())),frame=4)



actions = []
rewards = []

for ep in range(10000):
    img = env.reset()
    state=None

    episodic_reward = 0.0
    buffer=[]
    while True:
        previous_state = state
        action = env.action_space.sample()
        actions.append(action)
        state, reward, done, info = env.step(action)
        print(ep, state[0].shape,state[1], state[3], env.action_set[action], reward, done)

        rewards.append(reward)
        episodic_reward += reward
        if previous_state is not None:
            buffer.append([previous_state,action,reward,state,done])
        if done:
            time.sleep(20)
            break
    if len(buffer)==0:
        continue
    print('')
    print('Ep : ',ep, '    Return:',episodic_reward)
    writer.add_scalar('episodic_reward',episodic_reward,ep)
    writer.add_histogram('action',actions, ep)
    writer.add_histogram('reward',rewards, ep)
    with open('buffers/{}.pkl'.format(ep),'wb') as w:
        pickle.dump(buffer,w)



