"""
softmax 回归 · 从零实现（手打模板）

手打时的优先级：
  ★★★ 第 2 节 核心算法      —— 必须闭卷手打，这是全部价值所在（约 20 行）
  ★★  第 4 节 训练循环      —— 手打一遍，重点是 backward / updater / 梯度清零的顺序
  ★   第 3 节 评估          —— 手打一遍即可
  —   第 5 节 撞坑实验      —— 抄下来跑，不用背

对照 d2l 3.6：https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html
"""

import torch
from d2l import torch as d2l


# ==================== 0. 数据 ====================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# ==================== 1. 参数 ====================
num_inputs, num_outputs = 784, 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# ==================== 2. 核心算法 ★必须手打 ====================

def softmax(X):
    """X: (batch, num_outputs) -> 同形状，每行和为 1

    ⚠️ 这是书上的版本，有溢出问题。见第 5 节实验一。
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)   # (batch, 1)，靠广播相除
    return X_exp / partition


def net(X):
    """X: (batch, 1, 28, 28) -> (batch, 10)"""
    return softmax(X.reshape(-1, W.shape[0]) @ W + b)


def cross_entropy(y_hat, y):
    """y_hat: (batch, 10) 概率  y: (batch,) 真实类别下标 -> (batch,) 每样本 loss

    y_hat[range(n), y] 是花式索引：第 i 行取第 y[i] 列，
    也就是「模型给正确类别打的概率」。
    """
    return -torch.log(y_hat[range(len(y_hat)), y])


def sgd(params, lr, batch_size):
    """小批量随机梯度下降

    两个必须理解的点：
      1. no_grad  —— 参数更新本身不能被记进计算图
      2. zero_()  —— 不清零梯度会累加，训练直接崩（见实验三）
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ==================== 3. 评估 ====================

def accuracy(y_hat, y):
    """返回预测正确的样本数（float）"""
    preds = y_hat.argmax(dim=1)
    return float((preds.type(y.dtype) == y).sum())


@torch.no_grad()
def evaluate_accuracy(net, data_iter):
    correct, total = 0.0, 0
    for X, y in data_iter:
        correct += accuracy(net(X), y)
        total += y.numel()
    return correct / total


# ==================== 4. 训练循环 ====================

def train_epoch(net, train_iter, loss, updater):
    """返回 (平均loss, 训练精度)"""
    sum_loss, sum_correct, total = 0.0, 0.0, 0

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)          # (batch,)，没有 reduction

        l.sum().backward()          # 对 sum 求导
        updater(X.shape[0])         # updater 内部再除以 batch_size

        sum_loss += float(l.sum())
        sum_correct += accuracy(y_hat, y)
        total += y.numel()

    return sum_loss / total, sum_correct / total


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """每个 epoch 都打印，能看到收敛过程"""
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
    d2l.plt.show()          # ← 脚本里必须有这句，否则不显示


# ==================== 5. 撞坑实验 ====================

def softmax_stable(X):
    """数值稳定版：先减去每行最大值（log-sum-exp trick）"""
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    return X_exp / X_exp.sum(1, keepdim=True)


def experiment_1_overflow():
    """实验一：softmax 溢出"""
    print('\n=== 实验一：softmax 溢出 ===')
    X = torch.tensor([[100.0, 1000.0]])
    print('朴素版 :', softmax(X))          # tensor([[nan, nan]])
    print('稳定版 :', softmax_stable(X))   # tensor([[0., 1.]])
    print('→ exp(1000) 超出 float32 上限变成 inf，inf/inf = nan')
    print('→ nn.CrossEntropyLoss 内部做的就是稳定版，所以它接收 logits 而不是概率')


def experiment_2_no_grad():
    """实验二：更新参数时不加 no_grad"""
    print('\n=== 实验二：忘记 no_grad ===')
    p = torch.ones(2, requires_grad=True)
    (p.sum() * 2).backward()
    try:
        p -= 0.1 * p.grad          # 没有 no_grad
    except RuntimeError as e:
        print('报错:', str(e)[:80])
    print('→ 原地修改一个 requires_grad 的叶子张量，autograd 不允许')


def experiment_3_no_zero():
    """实验三：不清零梯度"""
    print('\n=== 实验三：忘记 zero_() ===')
    p = torch.ones(1, requires_grad=True)
    for step in range(3):
        (p * 2).sum().backward()   # 每次真实梯度都是 2
        print(f'  第 {step + 1} 次 backward 后 p.grad = {p.grad.item()}')
    print('→ 梯度在累加：2, 4, 6...  实际训练里步长会越来越大，loss 直接发散')


# ==================== main ====================

if __name__ == '__main__':
    lr = 0.1
    num_epochs = 10

    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    print(f'训练前随机精度: {evaluate_accuracy(net, test_iter):.4f}  (应该 ≈ 0.1)\n')

    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    experiment_1_overflow()
    experiment_2_no_grad()
    experiment_3_no_zero()

    predict(net, test_iter)
