import torch
from d2l import torch as d2l

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

from torch.utils import data #这里忘了

def load_array(data_arrays, batch_size, is_train=True):
    data_set = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = is_train
    )

batch_size = 10
data_iter = load_array((features, labels), batch_size)

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.zeros_(net[0].bias)

loss = nn.MSELoss()
trainer = torch.optim.SGD(
    net.parameters(),
    lr=0.03
)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    with torch.no_grad():
        train_l = loss(net(features), labels)

    print(f"epoch {epoch + 1}, loss {float(train_l):f}")

print(net[0].weight.data)
print(net[0].bias.data)

