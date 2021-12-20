from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.utils import save_image
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
import torchvision.models as models
from torchvision import transforms
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
#
from custom_datasets import *
from custom_transforms import *
from utils import *
from model import *


def get_args():
    parser = argparse.ArgumentParser(description='AutoDO using Implicit Differentiation')
    parser.add_argument('--data', default='./local_data', type=str, metavar='NAME',
                        help='folder to save all data')
    parser.add_argument('--dataset', default='MNIST', type=str, metavar='NAME',
                        help='dataset MNIST/CIFAR10/CIFAR100/SVHN/SVHN_extra/ImageNet')
    parser.add_argument('--workers', default=4, type=int, metavar='NUM',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, metavar='NUM',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=str, default='150,175,195', metavar='LR',
                        help='learning rate decay epochs (default: 150,175,195')
    parser.add_argument('--lr-warm-epochs', type=int, default=5, metavar='LR',
                        help='number using cosine annealing (default: False')
    parser.add_argument("--gpu", default='0', type=str, metavar='NUM',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=500, metavar='NUM',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-debug', action='store_true', default=False,
                        help='plot train images for debugging purposes')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of [1:C/2] to [C/2+1:C] labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-sr', '--subsample-ratio', type=float, default=1.0, metavar='N',
                        help='ratio of selected to total labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noise-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy (randomly flipped) labels (default: 0.0)')
    parser.add_argument('-r', '--run-folder', default='run0', type=str,
                        help='dir to save run')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='ablation: estimate DA from test data (default: False)')
    parser.add_argument('--oversplit', action='store_true', default=False,
                        help='ablation: train on all data (default: False)')
    parser.add_argument('--aug-model', default='NONE', type=str,
                        help='type of augmentation model NONE/RAND/AUTO/DADA/SHAred/SEParate parameters (default: NONE)')
    parser.add_argument('--los-model', default='NONE', type=str,
                        help='type of model for other loss hyperparams NONE/SOFT/WGHT/BOTH (default: NONE)')
    parser.add_argument('--hyper-opt', default='NONE', type=str,
                        help='type of bilevel optimization NONE/HES (default: NONE)')
    parser.add_argument('--hyper-steps', type=int, default=0, metavar='NUM',
                        help='number of gradient calculations to achieve grad(L_train)=0 (default: 0)')
    parser.add_argument('--hyper-iters', type=int, default=5, metavar='NUM',
                        help='number of approxInverseHVP iterations inside hyperparameter estimation loop (default: 5)')
    parser.add_argument('--hyper-alpha', type=float, default=0.01, metavar='HO',
                        help='hyperparameter learning rate (default: 0.01)')
    parser.add_argument('--hyper-beta', type=int, default=0, metavar='HO',
                        help='hyperparameter beta (default: 0)')
    parser.add_argument('--hyper-gamma', type=int, default=0, metavar='HO',
                        help='hyperparameter gamma (default: 0)')

    args = parser.parse_args()

    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    args.lr_warm = True
    args.lr_cosine = True
    dataset = args.dataset
    run_folder = args.run_folder
    # create folders
    if not os.path.isdir(args.data):
        os.mkdir(args.data)
    save_folder = '{}/{}'.format(args.data, dataset)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    long_run_folder = '{}/{}'.format(save_folder, run_folder)
    print('long_run_folder:', long_run_folder)
    if not os.path.isdir(long_run_folder):
        os.mkdir(long_run_folder)
    model_folder = '{}/{}'.format(save_folder, run_folder)
    print('model_folder:', model_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    # shared among datasets
    task_optimizer = 'sgd'
    task_momentum = 0.9
    task_weight_decay = 0.0001
    task_nesterov = True
    aug_mode = 0
    #

    if dataset == 'CIFAR10':
        total_images = 50000
        valid_images = 10000
        train_images = total_images - valid_images
        num_classes = 10
        num_channels = 3
        # data:
        test_data = CIFAR10(save_folder, train=False, transform=transforms.ToTensor(), download=True)
        train_data = CIFAR10(save_folder, train=True, transform=transforms.ToTensor(), download=True)

        task_lr = 0.1
        train_batch_size = 256
        hyper_batch_size = 256
        args.hyper_theta = ['cls']

    elif dataset == 'SVHN':
        num_classes = 10
        num_channels = 3
        test_data = SVHN(save_folder, split='test', transform=transforms.ToTensor(), download=True)
        train_data = SVHN(save_folder, split='train', transform=transforms.ToTensor(), download=True)
        total_images = 73257
        valid_images = 23257
        train_images = total_images - valid_images
        task_lr = 0.005
        train_batch_size = 256
        hyper_batch_size = 256
        args.hyper_theta = ['cls']

    elif dataset == 'ImageNet':
        total_images = 1281167
        valid_images = int(0.2 * total_images)  # 20% of train
        train_images = total_images - valid_images
        num_classes = 1000
        num_channels = 3
        test_data = ImageNet(save_folder, split='val', transform=transforms.ToTensor(), download=False)
        train_data = ImageNet(save_folder, split='train', transform=transforms.ToTensor(), download=False)
        print('TRANSFORM:', transform_train_clearimagenet)
        task_lr = 0.01
        train_batch_size = 256
        hyper_batch_size = 128
        args.hyper_theta = ['cls']
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(dataset))

    # dataloaders:
    data_file = '{}.pt'.format(model_folder)
    if os.path.isfile(data_file):
        valid_sub_indices, train_sub_indices, train_targets = torch.load(data_file)  # load saved indices
    else:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=valid_images, random_state=0)
        sss = sss.split(list(range(total_images)), train_data.targets)
        for _ in range(random.randint(1, 5)):
            train_indices, valid_indices = next(sss)
        #
        train_indices, valid_indices = list(train_indices), list(valid_indices)
        valid_sub_indices = valid_indices
        # save targets for soft label estimation
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False, **kwargs)
        MLEN = len(train_loader.dataset)  # dataset size
        BLEN = len(train_loader)  # number of batches
        train_targets = torch.zeros(MLEN, dtype=torch.long)
        for batch_idx, data in enumerate(train_loader):
            if batch_idx % args.log_interval == 0:
                print('Reading train batch {}/{}'.format(batch_idx, BLEN))
            _, train_target, train_index = data
            train_targets[train_index] = train_target
        # subsampling
        SR = int(1.0 * train_images * subsample_ratio)  # number of subsampled examples
        train_sr_indices = random.sample(train_indices, SR)
        #
        train_sub_data = torch.utils.data.Subset(train_data, train_sr_indices)
        train_sub_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=False,
                                                       **kwargs)
        SUB = len(train_sub_loader.dataset)
        print('Train dataset/subset: {}->{}'.format(MLEN, SUB))
        # imbalance
        train_sub_indices = train_sr_indices  # use all train subsampled data

        # save indices
        with open(data_file, 'wb') as f:
            torch.save((valid_sub_indices, train_sub_indices, train_targets), f)
    # samplers
    print('Valid/Train Split: {}/{}'.format(len(valid_sub_indices), len(train_sub_indices)))
    # loaders
    train_sub_data = torch.utils.data.Subset(train_data, train_sub_indices)
    valid_sub_data = torch.utils.data.Subset(train_data, valid_sub_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_batch_size, shuffle=False, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_sub_data, batch_size=hyper_batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=train_batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
    hyper_loader = torch.utils.data.DataLoader(train_sub_data, batch_size=hyper_batch_size, shuffle=True,
                                                   drop_last=True, **kwargs)
    # train data augmentation model

    # save other hyperparameters to arguments
    args.lr = task_lr
    args.train_batch_size = train_batch_size
    args.num_classes = num_classes
    # task optimizer
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list()
    for i in iterations:
        args.lr_decay_epochs.append(int(i))
    if args.lr_warm:
        args.lr_warmup_from = args.lr / 10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                        1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr

    writer = SummaryWriter('./logs')
    classifier = resnet18(pretrained=True).to(device)

    vae = VAE().cuda()
    optimizer = optim.SGD(list(classifier.parameters()) + list(vae.parameters()), lr=args.lr, momentum=task_momentum,
                          weight_decay=task_weight_decay,
                          nesterov=task_nesterov)

    apply = True
    best_acc = 0.0
    for epoch in range(0, args.epochs):
        vae.train()
        classifier.train()
        print('Run {}: {:.0f}% ({}/{})'.format(model_folder, 100.0 * epoch / args.epochs, epoch, args.epochs))
        adjust_learning_rate(args, optimizer, epoch)

        if epoch % 50 == 0:
            if apply:
                for param in vae.parameters():
                    param.requires_grad = True
                for param in classifier.parameters():
                    param.require_grad = False
            else:
                for param in vae.parameters():
                    param.requires_grad = False
                for param in classifier.parameters():
                    param.require_grad = True

        BN = len(train_loader)
        VN = len(valid_loader)
        N = len(train_loader.dataset)
        train_loss = 0.0
        validation_loss = 0.0
        acc_t = 0
        acc_v = 0

        for batch_idx, data in enumerate(train_loader):
            image = data[0].to(device)
            B, C, H, W = image.shape
            target = data[1].to(device)
            lr = warmup_learning_rate(args, epoch, batch_idx, BN, optimizer)

            xre, mu, logvar = vae(image, device)
            classification = classifier(xre)
            prediction = torch.argmax(classification, dim=1)

            if epoch % 2 == 0 and batch_idx == 10:
                for i in range(B):
                    if i % 50 == 0:
                        save_image(xre[i],
                                   "/home/nengyw/autodo/images/image_run9/epoch {} image {} prediction {}.png".format(
                                       epoch, i, prediction[i]))
                        save_image(image[i],
                                   "/home/nengyw/autodo/images/image_run9/epoch {} image {} ground truth {}.png".format(
                                       epoch, i, target[i]))
            acc_training = torch.sum(prediction == target) / B
            acc_t += acc_training
            celoss = F.cross_entropy(classification, target).to(device)
            kdloss = vae.loss(mu, logvar, device)
            loss = celoss + 0.1 * kdloss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_t /= BN
        train_loss /= BN
        print('Epoch: {}  training loss: {} training acc: {}'.format(epoch, train_loss, acc_t))

        if 1 == 1:
            classifier.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(valid_loader):
                    image = data[0].to(device)
                    target = data[1].to(device)
                    index = data[2].to(device)
                    expectation = classifier(image)
                    losss = F.cross_entropy(expectation, target).to(device)
                    losss = losss.item()
                    expectation = torch.argmax(expectation, dim=1)
                    acc = (torch.sum(expectation == target)) / 256
                    validation_loss = validation_loss + losss
                    acc_v += acc
                validation_loss /= VN
                acc_v /= VN
                print("Epoch: {} valid Loss: {} valid Acc {}".format(epoch, validation_loss, acc_v))
        if (epoch + 1) % 50 == 0:
            apply = (apply + 1) % 2

        writer.add_scalars('loss', {
            'Train_loss': train_loss,
            'Valid_loss': validation_loss,
        }, epoch)

        writer.add_scalars('accuracy', {
            'Train_Accuracy': acc_t,
            'Valid_Accuracy': acc_v,
        }, epoch)
        writer.add_scalar('Training_Accuracy', acc_t, epoch)
        writer.add_scalar('Training_loss', train_loss, epoch)
        writer.add_scalar('Validation_Accuracy', acc_v, epoch)
        writer.add_scalar('Validation_loss', validation_loss, epoch)

    writer.close()


if __name__ == '__main__':
    args = get_args()
    main(args)

