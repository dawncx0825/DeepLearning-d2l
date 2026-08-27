"""
参考示例：把 MLP 从 nn.Sequential 改写成 nn.Module 子类
========================================================
这份是给你看的，不是今天的作业。
今天的作业是自己写 mlp_ckpt.py（nn.Module + 训练循环 + checkpoint + 设备管理）。

这里只讲 nn.Module 这一层，训练循环和 checkpoint 留给你。
直接跑：python mlp_module_example.py
"""

import torch
from torch import nn

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# =============================================================
# 一、你已经会的写法
# =============================================================
net_seq = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)


# =============================================================
# 二、改写成 nn.Module 子类
# =============================================================
class MLP(nn.Module):
    """两层 MLP。和上面的 net_seq 完全等价，只是把结构写开了。

    改写只有两件事：
      __init__  声明用到哪些层（层本身是有参数的对象，要存下来）
      forward   规定数据怎么从这些层里穿过去
    """

    def __init__(self, num_inputs=784, num_hiddens=256, num_outputs=10):
        # super().__init__() 必须在第一行。
        # nn.Module 在这里建好 _parameters / _modules 这些登记簿，
        # 后面每次 self.xxx = 某个层，都是往登记簿里记一笔。
        # 忘了写会直接报错：cannot assign module before Module.__init__() call
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.act = nn.ReLU()
        self.out = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        # 形状表（B = batch size）：
        #   输入      (B, 1, 28, 28)
        X = self.flatten(X)   # (B, 784)
        X = self.hidden(X)    # (B, 256)
        X = self.act(X)       # (B, 256)   激活不改形状
        return self.out(X)    # (B, 10)    logits，没过 softmax
        # 不在这里加 softmax。CrossEntropyLoss 内部自带 log_softmax，
        # 你再加一次等于做了两遍，loss 会变得很奇怪但不报错。


# =============================================================
# 三、⚠️ 静默失败示范：参数没注册
# =============================================================
class BrokenMLP(nn.Module):
    """能实例化、能前向、能训练，但 loss 永远不动。一个错误提示都没有。"""

    def __init__(self):
        super().__init__()
        # 用 python list 装层 —— nn.Module 的 __setattr__ 拦不住 list，
        # 里面的 Linear 一个都没进登记簿
        self.layers = [nn.Linear(784, 256), nn.Linear(256, 10)]

    def forward(self, X):
        X = X.flatten(1)
        return self.layers[1](torch.relu(self.layers[0](X)))


class FixedMLP(nn.Module):
    """改法一：nn.ModuleList —— 会注册，但不定义 forward，顺序还得自己写。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 256), nn.Linear(256, 10)])

    def forward(self, X):
        X = X.flatten(1)
        return self.layers[1](torch.relu(self.layers[0](X)))


# 改法二：直接 self.a = nn.Linear(...)，就是上面 MLP 的写法
# 改法三：self.net = nn.Sequential(...)，既注册也定义 forward


# =============================================================
# 四、初始化
# =============================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
    # apply 会递归走遍所有子模块，包括 Flatten / ReLU，
    # 所以必须用 isinstance 判断，否则会往没有 weight 的层上乱设


# =============================================================
# 五、自检
# =============================================================
if __name__ == "__main__":
    print(f"torch {torch.__version__} | mps available: "
          f"{torch.backends.mps.is_available()} | device: {device}")
    print()

    net = MLP()
    net.apply(init_weights)

    # --- 1. 参数量：唯一可信的「注册成功」证据 ---
    n = lambda m: sum(p.numel() for p in m.parameters())
    print("参数量")
    print(f"  MLP        {n(net):>7}   ← 784*256+256 + 256*10+10 = 203530")
    print(f"  Sequential {n(net_seq):>7}   ← 应该一样")
    print(f"  BrokenMLP  {n(BrokenMLP()):>7}   ← 0，optimizer 会拿到空参数组")
    print(f"  FixedMLP   {n(FixedMLP()):>7}")
    print()

    # --- 2. 注册了什么 ---
    print("注册的子模块:", list(dict(net.named_children()).keys()))
    print("注册的参数  :", [k for k, _ in net.named_parameters()])
    print("BrokenMLP 的:", [k for k, _ in BrokenMLP().named_parameters()], "← 空")
    print()

    # --- 3. 逐层形状检查（形状出问题时先跑这个）---
    print("逐层形状")
    X = torch.randn(4, 1, 28, 28)
    print(f"  {'input':10} {tuple(X.shape)}")
    for name, layer in net.named_children():
        X = layer(X)
        print(f"  {name:10} {tuple(X.shape)}")
    print()

    # --- 4. 设备：搬之后才算数 ---
    net = net.to(device)
    print("设备")
    print(f"  device 变量     : {device}")
    print(f"  参数实际所在     : {next(net.parameters()).device}   ← 这行才是证据")
    print()

    # --- 5. 调用用 net(X)，不要用 net.forward(X) ---
    #     net(X) 走的是 __call__，里面还挂着 hook 机制；
    #     直接调 forward 能出结果，但 hook 全部失效，也不报错。
    y = net(torch.randn(4, 1, 28, 28, device=device))
    print(f"输出形状 {tuple(y.shape)}  设备 {y.device}")


# =============================================================
# 接下来轮到你：mlp_ckpt.py
# =============================================================
# 1. 训练循环自己写，别用 d2l.train_ch3 —— 你得控制 checkpoint 插在哪
# 2. 每个 epoch 末尾存一次，至少存四样：
#       model.state_dict()  optimizer.state_dict()  epoch  loss（记得 .item()）
# 3. 跑 3 个 epoch 退出 → 重新运行 → 从 checkpoint 接着训
#
# 验收：epoch 4 的 loss 紧接着 epoch 3 往下走，不能跳回初始值附近
#
# 这些都不报错，自己防：
#   □ 存了带计算图的 loss 而不是 loss.item()
#   □ optimizer 状态没存（SGD 看不出来，换 Adam 动量全丢）
#   □ epoch 编号没存 → 恢复后从 0 重跑，你看 loss 在降以为对了，其实是重训
#   □ 恢复后忘了 net.train()
#   □ torch.load 没给 map_location
