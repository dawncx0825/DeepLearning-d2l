import torch
import os
from torch import nn
from d2l import torch as d2l

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

class MLP(nn.Module):
    def __init__(self, num_inputs = 784, num_hidden = 256, num_outputs = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(num_inputs, num_hidden)
        self.act = nn.ReLU()
        self.out = nn.Linear(num_hidden, num_outputs)

    def forward(self, X):
        X = self.flatten(X)
        X = self.hidden(X)
        X = self.act(X)
        X = self.out(X)
        return X

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)

@torch.no_grad()
def evaluate(net, loss, test_iter):
    net.eval()
    sum_loss, acc, tot = 0, 0, 0
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        sum_loss += loss(y_hat, y).item() * len(X)
        acc += (y_hat.argmax(dim=1) == y).sum().item()
        tot += len(X)
    return sum_loss / tot, acc / tot

def train(net, loss, train_iter, test_iter, optimizer, path, num_epochs, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        ave_loss, ave_acc = evaluate(net, loss, test_iter)
        print(f"epoch:{epoch + 1} loss:{ave_loss:.4f} acc:{ave_acc:.4f}" )

        save_checkpoint(path, net, optimizer, epoch)
            
def save_checkpoint(path, net, optimizer, epoch):
    torch.save({
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
    }, path)


def load_checkpoint(path, net, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']

if __name__ == "__main__":
    net = MLP()
    net.apply(init_weights)
    net.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    num_epochs = 15

    path = 'mlp_checkpoint.pth'
    if os.path.exists(path):
        start_epoch = load_checkpoint(path, net, optimizer, device)
    else:
        start_epoch = 0

    train(net, loss, train_iter, test_iter, optimizer, path, num_epochs, start_epoch)

