import torch
from d2l import torch as d2l

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input, num_output = 784, 10

W = torch.normal(0, 0.01, (num_input, num_output), requires_grad=True)
b = torch.zeros(num_output,requires_grad=True)

def softmax(X):
    X = torch.exp(X)
    sumX = X.sum(dim=1)
    sumX = sumX.reshape(-1, 1)
    return X / sumX

def net(X):
    X = torch.flatten(X, 1)
    y = torch.matmul(X, W) + b
    return softmax(y)

def cross_what_loss(y_hat, y):
    arr = list(range(len(y_hat)))
    return -torch.log(y_hat[arr, y])

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def accuracy(y_hat, y):
    now = torch.argmax(y_hat, dim=1)
    return float((now == y).sum())

def evaluate_accuracy(net, data_iter):
    with torch.no_grad():
        acc, tot = 0, 0
        for X, y in data_iter:
            acc += accuracy(net(X), y)
            tot += len(X)

        return acc / tot

def train_epoch(net, train_iter, loss, updater):
    sum_loss, acc, tot = 0, 0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        updater(len(X))
        sum_loss += float(l.sum())
        acc += accuracy(y_hat, y)
        tot += len(X)

    return sum_loss / tot, acc / tot

def train(net, train_iter, test_iter, num_epochs, loss, updater):
    print(f"epoch        aver_loss    train_acc    test_acc")
    for epoch in range(num_epochs):
        aver_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        print(f"{epoch + 1}    {aver_loss:.4f}    {train_acc:.3f}    {test_acc:.3f}")

if __name__ == "__main__":
    lr = 0.1
    num_epochs = 10
    def updater(batch_size):
        sgd((W, b), lr, batch_size)

    print(evaluate_accuracy(net, test_iter))

    train(net, train_iter, test_iter, num_epochs, cross_what_loss, updater)

