import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_data
from model import MLP
from utils import *

"""
Here we regress 2 values, sin(x) and cos(x).

We assume the network spits out values in radians,
so we compute the angle using these 2 values (atan2)
and calculate the loss using MSE.
"""

def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        out = model(data)
        sin, cos = out.transpose(1, 0)
        angle = torch.atan2(sin, cos)
        loss = F.mse_loss(angle, target)
        mea = torch.mean(torch.abs(target - angle))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print("\t{}/{}: loss: {:.3f} - mea: {:.3f}".format(
                i+1, num_iters, loss.item(), mea.item())
            )


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        sin, cos = out.transpose(1, 0)
        angle = torch.atan2(sin, cos)
        return torch.mean(torch.abs(angle - target))



def main():
    X_train, X_test, y_train, y_test = generate_data(500)

    # potential data preprocessing here

    net = MLP(3, 2, 8, 2, 'relu')
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    train(net, optim, X_train, y_train, int(1e4))

if __name__ == '__main__':
    main()
