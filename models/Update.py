#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.distributed.distributed_c10d import get_world_size
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import random
from sklearn import metrics
from dgc.dgc import DGC
from models.test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, world_size, rank):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        batch_loss = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = Variable(images.to(self.args.device)), Variable(labels.to(self.args.device))
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            batch_loss += loss
            loss.backward()
            # dgc.gradient_update()
            optimizer.step()
            # if self.args.verbose and rank == 0 and batch_idx % 10 == 0:
            #     print('Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         batch_idx, batch_idx * len(images), len(self.ldr_train.dataset),
            #                 100. * batch_idx / len(self.ldr_train), loss.item()))
        return batch_loss/len(self.ldr_train)

    def inference(self, net, dataset_test, idxs):
        dataset = DatasetSplit(dataset_test, idxs)
        self.args.verbose = False
        acc_test, loss_test = test_img(net, dataset, self.args)
        return acc_test, loss_test