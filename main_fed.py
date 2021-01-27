#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import os
from torch.multiprocessing import Process, Array
import pickle

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from dgc.dgc import DGC

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

def partition_dataset():
    """
    :return
    dataset_train: dataset
    dict_users: list of data index for each user. e.g. 100 indexes with 600 data points index
    """
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users

def build_model():
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob

def init_processing(rank, size, fn, lost_train, acc_train, epoch, dataset_train, idx, net_glob, backend='gloo'):
    """initiale each process by indicate where the master node is located(by ip and port) and run main function
    :parameter
    rank : int , rank of current process
    size : int, overall number of processes
    fn : function, function to run at each node
    backend : string, name of the backend for distributed operations
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size, loss_train, acc_train, epoch, dataset_train, idx, net_glob)

def run(rank, world_size, loss_train, acc_train, epoch, dataset_train, idx, net_glob):
    net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    dgc_trainer = DGC(model=net_glob, rank=rank, size=world_size, momentum=args.momentum, full_update_layers=[4], percentage=args.dgc)
    dgc_trainer.load_state_dict(torch.load('dgc_state_dict.pt'))

    epoch_loss = torch.zeros(1)
    for iter in range(args.local_ep):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=idx) #create LocalUpdate class
        b_loss = local.train(net=net_glob, world_size=world_size, rank=rank) #train local
        epoch_loss += b_loss
        if rank == 0:
            print("Local Epoch: {}, Local Epoch Loss: {}".format(iter, b_loss))
    dgc_trainer.gradient_update()
    epoch_loss /= args.local_ep
    dist.reduce(epoch_loss, 0, dist.ReduceOp.SUM)

    net_glob.eval()
    train_acc = torch.zeros(1)
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=idx) #create LocalUpdate class
    acc, loss = local.inference(net_glob, dataset_train, idx)
    train_acc += acc
    dist.reduce(train_acc, 0, dist.ReduceOp.SUM)

    if rank == 0:
        torch.save(net_glob.state_dict(), 'net_state_dict.pt')
        torch.save(dgc_trainer.state_dict(), 'dgc_state_dict.pt')
        epoch_loss /= world_size
        train_acc /= world_size
        loss_train[epoch] = epoch_loss[0]
        acc_train[epoch] = train_acc[0]
        print('Round {:3d}, Rank {:1d}, Average loss {:.6f}, Average Accuracy {:.2f}%'.format(epoch, dist.get_rank(), epoch_loss[0], train_acc[0]))


if __name__ == '__main__':
    dataset_train, dataset_test, dict_users = partition_dataset()
    net_glob = build_model().to('cpu')

    #toggle verbose
    args.verbose=True

    # copy weights
    torch.save(net_glob.state_dict(), 'net_state_dict.pt')
    # net_glob.load_state_dict(torch.load('net_state_dict.pt'))

    # training
    m = max(int(args.frac * args.num_users), 1)
    loss_train = Array('f', args.epochs)
    acc_train = Array('f', args.epochs)
    dgc_trainer = DGC(model=net_glob, rank=0, size=m, momentum=args.momentum, full_update_layers=[4], percentage=args.dgc)
    torch.save(dgc_trainer.state_dict(), 'dgc_state_dict.pt')


    for iter in range(args.epochs):
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #random set of m clients

        processes = []

        for i in range(m):
            idx = dict_users[idxs_users[i]]
            p = Process(target=init_processing, args=(i, m, run, loss_train, acc_train, iter, dataset_train, idx, net_glob))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # testing
    net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Avg Train Accuracy: {:.2f}".format(acc_train[-1]))
    print("Testing Accuracy: {:.2f}".format(acc_test))

    #Saving the objects train_loss and train_accuracy:
    file_name = "/save/pickle/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs)

    # os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # with open(file_name, "wb") as f:
    #     pickle.dump([loss_train, acc_train], f)

    #plot loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_train)), loss_train, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.dgc))

    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(acc_train)), acc_train, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.dgc))