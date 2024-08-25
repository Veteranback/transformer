import torch
import os
import tarfile
import shutil
import codecs
import math
from tqdm import tqdm




def get_positional_encoding(d_model, max_length=100):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

    return lr


def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = prefix + 'transformer.pth.tar'
    torch.save(state, filename)


def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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
