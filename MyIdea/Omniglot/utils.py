import os
import re
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

# Those two functions are taken from torchvision code because they are not available on pip as of 0.2.0
def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None

pass


def get_loss(prediction, labels):
    cross_entropy = nn.NLLLoss()
    return cross_entropy(prediction, labels)





def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def do_learning(net, optimizer, train_iter,iterations,Mydevice):
    """
    网络，优化器，数据集迭代器
    """
    net.train()
    for outer_loop in range(iterations):
        for i ,or_data in enumerate(train_iter):
            # Sample minibatch
            data, labels = or_data
            data,labels  = data.to(Mydevice),labels.to(Mydevice).view(-1)
            # Forward pass
            prediction = net(data)
            # Get loss
            loss = get_loss(prediction, labels)

            # Backward pass - Update fast net
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.data.item()