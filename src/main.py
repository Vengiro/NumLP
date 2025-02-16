from utils import *
from src.numpyNN import *
import numpy as np
from src.MLP import MLP
def DataTrain(epoch: int, lr: float, model: MLP, data_name: str):
    x_train, label_train, x_test, label_test = sample_data(data_name=data_name)

    loss_train = []
    loss_val = []
    epoch_list = np.arange(epoch)

    for i in range(epoch):
        l_train = model.train_epoch(x_train, label_train, lr)
        l_val = model.fit(x_test, label_test)
        loss_train.append(l_train)
        loss_val.append(l_val)
        print(f"Epoch: {i} Train Loss: {l_train} Test Loss: {l_val}")

    loss_train = np.array(loss_train)
    loss_train = loss_train.reshape(-1)
    loss_val = np.array(loss_val)
    loss_val = loss_val.reshape(-1)

    logs = {
        'train_loss': loss_train,
        'test_loss': loss_val,
        'epochs': epoch_list
    }

    plot_loss(logs)
    plt.show()
    plot_decision_boundary(x_train, label_train, model.forward)
    plt.show()
    plot_decision_boundary(x_test, label_test, model.forward)
    plt.show()

def D7(epoch: int, lr: float, model: MLP, x_train, label_train, x_test, label_test, f):

    loss_train = []
    loss_val = []
    epoch_list = np.arange(epoch)

    for i in range(epoch):
        l_train = model.train_epoch(x_train, label_train, lr)
        l_val = model.fit(x_test, label_test)
        loss_train.append(l_train)
        loss_val.append(l_val)
        print(f"Epoch: {i} Train Loss: {l_train} Test Loss: {l_val}")

    loss_train = np.array(loss_train)
    loss_train = loss_train.reshape(-1)
    loss_val = np.array(loss_val)
    loss_val = loss_val.reshape(-1)

    logs = {
        'train_loss': loss_train,
        'test_loss': loss_val,
        'epochs': epoch_list
    }

    plot_loss(logs)
    plt.show()
    plot_decision_boundary(x_train, label_train, model.forward, extra=True, f=f)
    plt.show()
    plot_decision_boundary(x_test, label_test, model.forward, extra=True, f=f)
    plt.show()


#D2
print("D2")
model = MLP(1, [(2, 10), (10, 1)], [relu, linear], L2, he, ADAM)
DataTrain(100, 0.01, model, 'linear-separable')

#D3
print("D3")
model = MLP(2, [(2, 32), (32, 16), (16, 1)], [relu, relu, linear], BCE, he, ADAM)
DataTrain(500, 0.01, model, 'XOR')

#D4
print("D4")
model = MLP(2, [(2, 16), (16, 8), (8, 1)], [relu, relu, sigmoid], L2, he, ADAM)
DataTrain(200, 0.03, model, 'circle')

model = MLP(2, [(2, 16), (16, 8), (8, 1)], [relu, relu, sigmoid], BCE, he, ADAM)
DataTrain(200, 0.01, model, 'circle')

#D5
print("D5")
model = MLP(5, [(2, 128), (128, 64), (64,32), (32, 16), (16, 8), (8, 1)],[relu, relu, relu, relu, relu, linear], L2, he, GD)
DataTrain(1000, 0.005, model, 'sinusoid')
model = MLP(5, [(2, 128), (128, 64), (64,32), (32, 16), (16, 8),  (8, 1)], [relu, relu, relu, relu, relu, linear], L2, he, momentumGD)
DataTrain(1000, 0.005, model, 'sinusoid')
model = MLP(5, [(2, 128), (128, 64), (64,32), (32, 16), (16, 8), (8, 1)], [relu, relu, relu, relu, relu, linear], L2, he, ADAM)
DataTrain(1000, 0.005, model, 'sinusoid')

#D6
print("D6")
model = MLP(5, [(2, 128), (128, 64), (64,32), (32, 16), (16, 8), (8, 1)], [relu, relu, relu, relu, relu, linear], L2, xavier, ADAM)
DataTrain(1000, 0.005, model, 'swiss-roll')

#D7
print("D7")
model = MLP(1, [(3, 10), (10, 1)], [relu, linear], L2, he, ADAM)
x_train, label_train, x_test, label_test = sample_data(data_name='circle')
x_tr_sq = np.sum(x_train ** 2, axis=1)
x_te_sq = np.sum(x_test ** 2, axis=1)
x_train = np.concatenate([x_train, x_tr_sq.reshape(-1, 1)], axis=1)
x_test = np.concatenate([x_test, x_te_sq.reshape(-1, 1)], axis=1)
D7(500, 0.005, model, x_train, label_train, x_test, label_test, f=lambda x, y: x ** 2 + y ** 2)

model = MLP(1, [(3, 10), (10, 1)], [relu, linear], L2, he, ADAM)
x_train, label_train, x_test, label_test = sample_data(data_name='XOR')
x_tr_mul = np.prod(x_train, axis=1).reshape(-1, 1)
x_te_mul = np.prod(x_test, axis=1).reshape(-1, 1)
x_train = np.concatenate([x_train, x_tr_mul], axis=1)
x_test = np.concatenate([x_test, x_te_mul], axis=1)
D7(500, 0.005, model, x_train, label_train, x_test, label_test, f=lambda x, y: x*y)

model = MLP(4, [(3, 64), (64,32), (32, 16), (16, 8), (8, 1)], [relu, relu, relu, relu, linear], L2, xavier, ADAM)
x_train, label_train, x_test, label_test = sample_data(data_name='swiss-roll')
x_tr_sq = np.sum(x_train ** 2, axis=1)
x_te_sq = np.sum(x_test ** 2, axis=1)
x_train = np.concatenate([x_train, x_tr_sq.reshape(-1, 1)], axis=1)
x_test = np.concatenate([x_test, x_te_sq.reshape(-1, 1)], axis=1)
D7(1000, 0.005, model, x_train, label_train, x_test, label_test, f=lambda x, y: x ** 2 + y ** 2)




