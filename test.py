import time

import numpy as np
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn

from base_model import CNN, DeCNN, Autoencoder
from matplotlib import pyplot as plt


from scipy.ndimage import imread
from datagen import DSDataLoader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i,X in enumerate(loader.load_batch()):
        # measure data loading time
        data_time.update(time.time() - end)
        #X_var = Variable(torch.from_numpy(X)).cuda() 
        X_var = Variable(torch.FloatTensor(X)).cuda() 

        # compute output
        output = model(X_var)
        loss = criterion(output, X_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], X_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
               epoch, i, len(loader), batch_time=batch_time,
               data_time=data_time, loss=losses)


def main():

    model = Autoencoder(encoder=CNN, decoder=DeCNN, hidden=100).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01,
                            momentum=0.9,
                            weight_decay=1e-4)

    datagen = DSDataLoader(batch_size=32)

    print 'Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))

    for epoch in range(0, 100):
        train(datagen, model, criterion, optimizer, epoch)

def test():
    model = Autoencoder(encoder=CNN, decoder=DeCNN, hidden=100).cuda()
    img = imread('/home/leesy714/dataset/ds/0.png', mode='RGB')
    img = img / 255.0
    img = np.array([img],dtype=np.float).transpose((0,3,1,2))
    X_var = Variable(torch.FloatTensor(img)).cuda() 
    out = model(X_var)
    print out
    
if __name__=='__main__':
    #test()
    main()
