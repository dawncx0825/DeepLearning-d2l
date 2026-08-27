import torch
x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
X = x.reshape(3, 4)
print(X)
print(X.shape)
print(torch.tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]]))
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))