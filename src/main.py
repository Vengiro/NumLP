from blackBox import *
from numpyNN import *
import numpy as np
def D2(epoch: int, lr: float):
    x_train, label_train = linearData(200)
    x_test, label_test = linearData(200)

    model = MLP(1, [(2, 10), (10, 1)], relu, L2)
    loss_train = []
    loss_val = []
    epoch_list = np.arange(epoch)

    for i in range(epoch):
        l_train = model.train_epoch(x_train, label_train, lr)
        l_val = model.fit(x_test, label_test)
        loss_train.append(l_train)
        loss_val.append(l_val)
        print(f"Epoch: {i} Train Loss: {l_train} Test Loss: {l_val}")

    logs = {
        'train_loss': loss_train,
        'test_loss': loss_val,
        'epochs': epoch_list
    }

    plot_loss(logs)
    plot_decision_boundary(x_test, label_test, model.forward)

D2(100, 0.01)