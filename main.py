import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import generate_data
from model import MLP
from utils import *


def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        logits = model(data)
        sin_pred, cos_pred = logits.transpose(1, 0)
        angle_pred = torch.atan2(sin_pred, cos_pred).unsqueeze_(1)
        loss = F.mse_loss(angle_pred, target)
        mae = torch.mean(torch.abs(target - angle_pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print("\t{}/{}: loss: {:.3f} - mae: {:.3f}".format(
                i+1, num_iters, loss.item(), mae.item())
            )
    print("\tMean Absolute Error (MAE): {:.3f}".format(mae))


def test(model, data, target, plot=True):
    with torch.no_grad():
        logits = model(data)
    sin, cos = logits.transpose(1, 0)
    pred = torch.atan2(sin, cos)
    mae = torch.mean(torch.abs(pred - target))
    print("\tMean Absolute Error (MAE): {:.3f}".format(mae))
    if plot:
        pred = pred.squeeze().numpy()
        true = target.squeeze().numpy()
        fig, ax = plt.subplots()
        ax.scatter(pred, true)
        ax.plot(
            true, true,
            alpha=0.4, linestyle='--',
            dashes=(3, 10), color='r',
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.grid()
        plt.show()


def main():
    X_train, X_test, y_train, y_test = generate_data(100, dims=2)

    net = MLP(2, 2, 4, 2, 'tanh')
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("Training...")
    train(net, optim, X_train, y_train, int(1e4))
    print("Testing...")
    test(net, X_test, y_test, True)


if __name__ == '__main__':
    main()
