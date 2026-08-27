import torch
from d2l import torch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


num_inputs, num_outputs = 784, 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(X.reshape(-1, W.shape[0]) @ W + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def accuracy(y_hat, y):
    preds = y_hat.argmax(dim=1)
    return float((preds.type(y.dtype) == y).sum())



def evaluate_accuracy(net, data_iter):
    with torch.no_grad():
        correct, total = 0.0, 0
        for X, y in data_iter:
            correct += accuracy(net(X), y)
            total += y.numel()
        return correct / total


def train_epoch(net, train_iter, loss, updater):
    sum_loss, sum_correct, total = 0.0, 0.0, 0

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        l.sum().backward()
        updater(X.shape[0])

        sum_loss += float(l.sum())
        sum_correct += accuracy(y_hat, y)
        total += y.numel()

    return sum_loss / total, sum_correct / total


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    print(f'{"epoch":>5} | {"train loss":>10} | {"train acc":>9} | {"test acc":>8}')
    print('-' * 44)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'{epoch + 1:>5} | {train_loss:>10.4f} | {train_acc:>9.3f} | {test_acc:>8.3f}')


def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))
    titles = [t + '\n' + p for t, p in zip(trues, preds)]
    d2l.show_images(X[:n].reshape((n, 28, 28)), 1, n, titles=titles[:n])
    d2l.plt.show()


def softmax_stable(X):
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    return X_exp / X_exp.sum(1, keepdim=True)

if __name__ == '__main__':
    lr = 0.1
    num_epochs = 10

    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    print(f'训练前随机精度: {evaluate_accuracy(net, test_iter):.4f}  (应该 ≈ 0.1)\n')

    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict(net, test_iter)