import random
import numpy as np
import torch

__all__ = ['init_seeds', 'warmup_learning_rate', 'adjust_learning_rate']

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.lr_warm and epoch < args.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (args.lr_warm_epochs * total_batches)
        lr = args.lr_warmup_from + p * (args.lr_warmup_to - args.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:

