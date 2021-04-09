from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from helper_tool import DataProcessing as DP
from data_S3DIS import S3DIS
from model_seg import Seg, get_loss
import numpy as np
from torch.utils.data import DataLoader
from util import seg_loss, IOStream
import sklearn.metrics as metrics

import time


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(S3DIS(5, args.num_points, partition='train'), num_workers=2,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS(5, args.num_points, partition='val'), num_workers=2,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    # train_data = S3DIS(5, args.num_points, partition='train')
    # test_data = S3DIS(5, args.num_points, partition='val')

    device = torch.device("cuda" if args.cuda else "cpu")

    class_weights = torch.from_numpy(DP.get_class_weights('S3DIS')).to(device)
    class_weights = class_weights.float()
    model = Seg(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)
    print('num_points:%s, batch_size:%s, %s' % (args.num_points, args.batch_size, args.test_batch_size))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = seg_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        # logits_ = []
        # label_ = []
        # for i in range(train_data.__len__()):
        #     data, label = train_data.__getitem__(i)
        #     data, label = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device).squeeze()
        #     data, label = torch.unsqueeze(data, 0), torch.unsqueeze(label, 0)
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = args.batch_size

            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            # logits_.append(logits)
            # label_.append(label)

            # if len(logits_) < batch_size:
            #     continue

            opt.zero_grad()
            # logits, label = torch.cat(logits_, dim=-1), torch.cat(label_, dim=-1)
            loss = get_loss(logits, label, class_weights)
            loss.backward()
            # logits_.clear()
            # label_.clear()
            opt.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true += label.cpu().numpy().tolist()[0]
            train_pred += preds.detach().cpu().numpy().tolist()[0]

        print('train total time is', total_time)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                    train_true,
                                                                                    train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                    train_true,
                                                                                    train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        if epoch % 5 == 0:
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            total_time = 0.0
            idx = 0
            # logits_ = []
            # label_ = []
            # for i in range(test_data.__len__()):
            #     data, label = test_data.__getitem__(i)
            #     data, label = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device).squeeze()
            #     data, label = torch.unsqueeze(data, 0), torch.unsqueeze(label, 0)
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = args.test_batch_size

                start_time = time.time()
                logits = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                # logits_.append(logits)
                # label_.append(label)

                # if len(logits_) < batch_size:
                #     continue

                # logits, label = torch.cat(logits_, dim=-1), torch.cat(label_, dim=-1)
                loss = get_loss(logits, label, class_weights)
                # logits_.clear()
                # label_.clear()

                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true += label.cpu().numpy().tolist()[0]
                test_pred += preds.detach().cpu().numpy().tolist()[0]

            print('test total time is', total_time)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = '*** Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  test_loss*1.0/count,
                                                                                  test_acc,
                                                                                  avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                print('save new best model acc: %s' % best_test_acc)
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(S3DIS(5, args.num_points, partition='val'), num_workers=2,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Seg(args).to(device)
    model = nn.DataParallel(model)

    print('loading model: %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)            # B, one_hot, N
        preds = logits.max(dim=1)[1]    # B, N

        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true.reshape(-1), test_pred.reshape(-1))
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true.reshape(-1), test_pred.reshape(-1))
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=1,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,     # Deleted
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
