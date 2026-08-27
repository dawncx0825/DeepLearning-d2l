import torch
import random

def create_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = create_data(true_w, true_b, 1000)

def data_iter(features, labels, batch_size):
    n = len(features)
    arr = list(range(n))
    random.shuffle(arr)
    for i in range(0, n, batch_size):
        x = torch.tensor(arr[i : i + batch_size])
        yield features[x], labels[x]

w = torch.normal(0, 1, (2, 1))
b = torch.zeros(1)
w.requires_grad = True
b.requires_grad = True

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squareloss(y_hat, y):
    return (y_hat- y.reshape(y_hat.shape)) ** 2 / 2

def SGD(params, lr, batch_size):
    with torch.no_grad():
        for para in params:
            para -= lr * para.grad / batch_size
            para.grad.zero_()

num_epochs = 3
lr = 0.03
batch_size = 10
net = linreg
loss = squareloss

for epoch in range(num_epochs):
    for X, y in data_iter(features, labels, batch_size):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        SGD((w, b), lr, batch_size)
        
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch:{epoch + 1} loss:{float(train_l.mean()):f}")
