# 动手学深度学习 — 跟读代码

《动手学深度学习》(Dive into Deep Learning, PyTorch 版) 的学习代码。

除了跟着书敲的部分，还包含一些自己扩展的内容：**默写复现**、**故意写错的对照实验**，以及书上没有的**完整训练工程流程**（checkpoint、设备管理）。

## 环境

在 macOS (Apple Silicon) 上开发，**没有 CUDA**。所有涉及设备的代码统一用：

```python
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
```

依赖：`torch`、`d2l`、`matplotlib`；`recurrent-neural-networks/gpt_token.py` 额外需要 `tiktoken`。

---

## 目录结构

### `preliminaries/` — 第 2 章 预备知识

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `ndarray.py` | 2.1 数据操作 | 张量创建、reshape、广播、索引 |
| `pandas_lessons.py` | 2.2 数据预处理 | 读 csv、处理缺失值、转张量 |
| `autograd.py` | 2.5 自动微分 | `backward()`、非标量求导、控制流下的梯度 |
| `probability.py` | 2.6 概率 | 多项分布采样与频率收敛 |

### `linear-networks/` — 第 3 章 线性神经网络

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `linear-regression.py` | 3.2 线性回归从零实现 | 手写 `data_iter` / `linreg` / `squareloss` / `SGD` |
| `linear-regression-concise.py` | 3.3 线性回归简洁实现 | `nn.Linear` + `DataLoader` |
| `image-classification-dataset.py` | 3.5 图像分类数据集 | Fashion-MNIST 加载与可视化 |
| `softmax-regression.py` | 3.6 softmax 回归从零实现 | 初版 |
| `softmax-regression-solo.py` | 3.6 | 不看书默写的第一版 |
| `softmax-regression-recall.py` | 3.6 | 不看书默写的第二版 |
| `softmax-regression-clean.py` | 3.6 | 整理版，**含撞坑实验**（见下） |
| `softmax-regression-concise.py` | 3.7 softmax 回归简洁实现 | `nn.CrossEntropyLoss` |

`softmax-regression-clean.py` 末尾有三个对照实验，都是"能跑但结果不对"的典型：

- `experiment_1_overflow` — 不做 log-sum-exp 平移时 `exp` 溢出成 `inf`/`nan`
- `experiment_2_no_grad` — 评估时忘记 `torch.no_grad()`
- `experiment_3_no_zero` — 忘记 `zero_grad()`，梯度累加

### `multilayer-perceptrons/` — 第 4 章 多层感知机

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `multilayer-perceptrons.py` | 4.2 MLP 从零实现 | 手写 `relu` 和两层网络 |
| `multilayer-perceptrons-concise.py` | 4.3 MLP 简洁实现 | `nn.Sequential` |
| `multilayer-perceptrons-module.py` | *书外扩展* | **完整训练工程流程** |

`multilayer-perceptrons-module.py` 是这一章的重点，书上没有：

- 把 MLP 改写成 `nn.Module` 子类
- `evaluate()` 独立成函数，`net.eval()` 和 `@torch.no_grad()` 封装在函数**内部**（放在调用方是常见的静默 bug 来源）
- MPS 设备管理
- `save_checkpoint` / `load_checkpoint`：保存模型参数、优化器状态、epoch
- **跨进程验证**：训 5 轮 → 退出进程 → 重新启动 → 从第 6 轮接续

### `deep-learning-computation/` — 第 5 章 深度学习计算

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `mlp_module_example.py` | 5.1 层和块 | `nn.Sequential` → `nn.Module` 的改写对照 |

含 `BrokenMLP` / `FixedMLP` 一组对照：把子层放进普通 Python list 会导致参数不注册，**不报错**，只是永远不更新。

### `recurrent-neural-networks/` — 第 8 章 循环神经网络

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `gpt_token.py` | 8.2 文本预处理（扩展） | 用 `tiktoken` 观察 GPT-4 的 BPE 分词行为 |

对比书上的词元化方案，看 `"the"` 和 `" the"`、数字串、中文分别被切成什么。

### `attention/` — 第 10 章 注意力机制和 Transformer

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `attention-scoring-functions.py` | 10.3 注意力评分函数 | 加性注意力、缩放点积注意力 |
| `multihead-attention.py` | 10.5 多头注意力 | `transpose_qkv` / `transpose_output` / 多头封装 |
| `transformer.py` | 10.7 Transformer | 完整实现 |

`transformer.py` 自底向上包含：

```
masked_softmax
transpose_qkv / transpose_output
DotProduct_Attention
Multihead_Attention
PositionalEncoding
PositionWiseFFN
AddNorm
EncoderBlock      →  TransformerEncoder
DecoderBlock      →  TransformerDecoder
EncoderDecoder
```

除 `masked_softmax` 外均为手写实现。

---

## 记录下来的静默失败

这个仓库最主要的用途是记录**能正常运行但结果不对**的错误。这类问题不抛异常、形状检查也过，是深度学习调试里最花时间的部分。

| 现象 | 后果 |
|---|---|
| `argmax` 不指定 `dim` | 在错误的轴上取最大值 |
| 忘记 `zero_grad()` | 梯度累加，等效学习率漂移 |
| 评估时忘记 `net.eval()` | Dropout / BatchNorm 仍处训练模式 |
| 子模块放进普通 Python list | 参数不注册，optimizer 不更新 |
| `optimizer.load_state_dict` 覆盖新设的 `lr` | 改了超参但没生效 |
| `repeat` 与 `repeat_interleave` 混用 | 形状相同，掩码全部错位 |
| `reshape` 前漏 `permute` | 形状合法，不同头/位置的数据被打乱 |
| 4D 张量上用 `transpose(1, 2)` | 交换了错误的两轴，应用 `transpose(-2, -1)` |
| 自注意力中 q/k 传混 | q、k 同源同形状，一路跑到底不报错 |
| `squeeze()` 不带参数 | `batch=1` 时把 batch 轴一并删掉 |
| dropout 作用在 logits 而非注意力权重上 | `eval()` 模式下两种写法输出完全相同，测试抓不到 |

最后一条最难发现：它只在训练时存在，任何在 `eval()` 下做的单元测试都无法暴露它。

---

## 一个有用的测试模式

`transpose_qkv` / `transpose_output` 这类形状变换，**形状检查完全无效**——写错时形状仍然合法。用互逆测试：

```python
X = torch.arange(2*10*32, dtype=torch.float32).reshape(2, 10, 32)
assert torch.equal(transpose_output(transpose_qkv(X, 4), 4), X)
```

用 `arange` 而非 `randn`：每个元素都不同，任何顺序错乱都会被 `equal` 抓到。

---

## 参考

- 书籍：[动手学深度学习](https://zh.d2l.ai/)
- 代码：[d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
