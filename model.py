import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


import base_model



class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.saved_actions = []
        self.rewards = []

class CNN(nn.Module):
    def __init__(self, target1, target2):
        super(CNN, self).__init__()

        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(5120, self.screen_feature_num)


        self.head1 = nn.Linear(self.screen_feature_num, target1)
        self.head2 = nn.Linear(self.screen_feature_num, target2)

        
    def forward(self, screen ):
        screen = F.avg_pool2d(screen, (2,2))
        features = F.relu(self.conv1(screen))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))
        features = F.relu(self.conv4(features))
        features = F.relu(self.conv5(features))
        features = F.relu(self.conv6(features))
        
        features = features.view(features.size(0),-1)
        features = F.relu(self.fc1(features))

        pred = self.head1(features), self.head2(features)

        return pred, features 



class DQN(nn.Module):
    def __init__(self, action, variables, pretrained=None):
        super(DQN, self).__init__()
        self.model = CNN(10, 10)
        self.variables = variables
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            self.model.load_state_dict(checkpoint['state_dict'])
        in_features = self.model.screen_feature_num
        self.value_head = nn.Linear(in_features + self.variables, action)

    def forward(self, screen, variables):
        pred, feature = self.model(screen)
        feature = torch.cat((feature, variables), 1)
        action_values = self.value_head(feature)
        
        return action_values

    def parameters(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p




class PolicyCNN(Policy):
    def __init__(self, action, variables, pretrained=None):
        super(PolicyCNN, self).__init__()
        self.model = CNN(target1=10, target2=10)
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            self.model.load_state_dict(checkpoint['state_dict'])

        in_features = self.model.screen_feature_num

        self.action_head = nn.Linear(in_features + variables, action)
        self.value_head = nn.Linear(in_features + variables, 1)
        self.softmax = nn.Softmax()

    def forward(self, screen, variables):
        pred, feature = self.model(screen)
        feature = torch.cat((feature, variables), 1)
        action_score = self.action_head(feature)
        state_values = self.value_head(feature)
        
        return self.softmax(action_score), state_values

    def parameters(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p



