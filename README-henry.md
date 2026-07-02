# Note

* Please run train_save_point_v1.py to train, this file is based on train.py and support to save model and checkpoint after each epoch if there is better accuracy score, and load saved model and checkpoint to avoid training from zero-position after relaunch the program.

* Before you start training, you should read the original README.md to get base background knowledge, and then read [Records_of_Running_History-henry.md](Records_of_Running_History-henry.md) to get know how to prepare training data and run training with Python 3.11.

* The [Code_Learning-henry.md](Code_Learning-henry.md)/[Code_Learning-henry-en_us.md](Code_Learning-henry-en_us.md) explains detail code-level steps of Transformer alogrithem.

# Known Issues

* Since the size of training data is very small, only 30K, so after 50 epoches the accuracy score of validation will stop to increase, you should use more bigger training data instead.

# Changes in train_save_point_v1.py
---

## train_save_point_v1.py 与 train.py 差异分析

### 1️⃣ 文件头文档字符串变更

**train_save_point_v1.py** 新增了详细的命令行示例，并加了 emoji 前缀：

```python
# 新增
"""
 python train_save_point_v1.py \
  -data_pkl m30k_deen_shr.pkl \
  -output_dir output \
  -b 32 \
  -epoch 30 \
  -warmup 20000 \
  -lr_mul 0.5 \
  -label_smoothing \
 -no_cuda
"""
```

### 2️⃣ 新增全局常量

```python
# 新增
MODEL_NAME = "./output/model.chkpt"
CHECKPOINT_PATH = MODEL_NAME
```

### 3️⃣ 训练函数 `train_epoch()` — 新增梯度裁剪

```python
# 在 loss.backward() 之后，step_and_update_lr() 之前新增
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)
```
目的：**避免学习率快速增长导致梯度爆炸**。

### 4️⃣ 训练函数 `train()` 签名变更

| train.py | train_save_point_v1.py |
|---|---|
| `def train(model, training_data, validation_data, optimizer, device, opt)` | `def train(model, training_data, validation_data, optimizer, device, opt, start_epoch=0, valid_losses=None)` |

新增了 `start_epoch` 和 `valid_losses` 两个参数，**支持从中断的 epoch 恢复训练**。

### 5️⃣ `train()` 内部变更

- **日志文件模式**：从 `'w'` 改为 `'a'`（追加模式），支持断点续训
- **写入表头**：仅当 `start_epoch == 0` 时才写入
- **valid_losses 初始化**：从 `valid_losses = []` 改为 `if valid_losses is None: valid_losses = []`
- **epoch 循环**：从 `range(opt.epoch)` 改为 `range(start_epoch, opt.epoch)`
- **checkpoint 字典格式**：

| train.py | train_save_point_v1.py |
|---|---|
| `'model': model.state_dict()` | `'model_state_dict': model.state_dict()` |
| 无 | `'optimizer_state_dict': optimizer._optimizer.state_dict()` |
| 无 | `'optimizer_n_steps': optimizer.n_steps` |
| 无 | `'valid_losses': valid_losses` |

- **保存路径**：从 `os.path.join(opt.output_dir, model_name)` 改为直接使用全局 `MODEL_NAME`

### 6️⃣ `main()` 中新增断点续训逻辑

train_save_point_v1.py 在创建完模型和优化器后，增加了**自动检测 checkpoint** 的功能：

```python
# ========= Auto Resume from Checkpoint =========#
start_epoch = 0
saved_losses = []
if os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    # 兼容旧格式 "model" vs "model_state_dict"
    transformer.load_state_dict(checkpoint.get("model_state_dict") or checkpoint["model"])
    # 恢复优化器状态
    optimizer._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer.n_steps = checkpoint["optimizer_n_steps"]
    start_epoch = checkpoint["epoch"] + 1
    # 恢复历史验证损失
    saved_losses = checkpoint.get("valid_losses", [])
```

### 7️⃣ MPS 设备支持（Apple Silicon）

```python
# train.py
device = torch.device('cuda' if opt.cuda else 'cpu')

# train_save_point_v1.py
device = torch.device('cuda' if opt.cuda else ("mps" if torch.backends.mps.is_available() else "cpu"))
```

支持 Apple Silicon Mac 的 MPS 加速。

### 8️⃣ 打印输出添加 emoji 前缀

几乎所有 `print()` 语句都添加了 emoji：

- `"⚠️ "` — 警告
- `"✅  "` — 成功/信息
- `"🔥 "` — 训练 epoch 开始

### 9️⃣ 代码风格变化

- 双引号 `"` 替代单引号 `'`
- 格式化方式更倾向于多行参数形式（black 风格）
- 默认 batch size 改为 `32`（原始为 `2048`）
- 默认 warmup steps 改为 `20000`（原始为 `4000`）
- 默认 `lr_mul` 改为 `0.5`（原始为 `2.0`）

---

### 总结：核心目的

**train_save_point_v1.py 最主要的变化是增加了断点续训（Checkpoint Resume）功能**，使得训练中断后可以从上次保存的 epoch 继续，而无需从头开始。伴随这一核心变更的还有：

1. **梯度裁剪** — 防止梯度爆炸
2. **优化器状态保存/恢复** — 保持学习率调度连续性
3. **历史验证损失恢复** — 确保 `save_mode='best'` 的正确性
4. **MPS 支持** — Apple Silicon 设备兼容

# Upgrade train with MLX for apple silicon platform
>[!tip] Plan from DeepSeek V4

---

## 将该 Transformer 项目升级到 MLX 的完整方案

### 项目结构概览

| 文件 | 作用 | MLX 改造难度 |
|---|---|---|
| Models.py | Transformer 主模型 + 位置编码 + Encoder/Decoder | 🔴 高 |
| Layers.py | EncoderLayer / DecoderLayer | 🟡 中 |
| SubLayers.py | MultiHeadAttention + FFN | 🔴 高 |
| Modules.py | Scaled Dot-Product Attention | 🟡 中 |
| Optim.py | 学习率调度包装器 | 🟢 低 |
| Translator.py | Beam Search 解码 | 🔴 高 |
| train.py | 训练入口 | 🔴 高 |
| translate.py | 推理入口 | 🟡 中 |
| preprocess.py | 数据预处理（spaCy/BPE） | 🟢 不变 |

---

### 核心 API 映射表

#### 1. 基础模块映射

| PyTorch | MLX |
|---|---|
| `torch.nn.Module` | `mlx.nn.Module` |
| `torch.nn.Linear(d_in, d_out, bias)` | `mlx.nn.Linear(d_in, d_out, bias)` |
| `torch.nn.Embedding(vocab, d_model, padding_idx=idx)` | `mlx.nn.Embedding(vocab, d_model)` ⚠️ 无 `padding_idx` 参数 |
| `torch.nn.LayerNorm(d_model, eps=1e-6)` | `mlx.nn.LayerNorm(d_model, eps=1e-6)` |
| `torch.nn.Dropout(p=0.1)` | `mlx.nn.Dropout(p=0.1)` |
| `nn.ModuleList([...])` | `列表 + mlx.nn.Sequential(...)` 或纯列表 |

#### 2. Tensor 操作映射

| PyTorch | MLX |
|---|---|
| `torch.tensor(...)` | `mx.array(...)` |
| `torch.zeros_like(x)` | `mx.zeros_like(x)` |
| `torch.ones((a,b))` | `mx.ones((a,b))` |
| `torch.FloatTensor(x)` | `mx.array(x, dtype=mx.float32)` |
| `torch.matmul(a, b)` | `a @ b` 或 `mx.matmul(a, b)` |
| `x.transpose(1,2)` | `mx.transpose(x, axes=(0,2,1,3))` |
| `x.contiguous().view(a,b)` | `x.reshape(a,b)` |
| `x.unsqueeze(1)` | `mx.expand_dims(x, axis=1)` |
| `x.eq(y)` | `x == y` |
| `x.ne(y)` | `x != y` |
| `.masked_select(mask)` | `x[mask]` |
| `.masked_fill(mask, val)` | `mx.where(mask, val, x)` |
| `.scatter_(dim, idx, src)` | 需手动实现或用 `mx.put_along_axis` |
| `torch.triu(x, diagonal=1)` | `mx.triu(x, k=1)` |
| `.sum()`, `.mean()` | 相同 |
| `.item()` | `.item()` |
| `.size(0)` | `.shape[0]` |

#### 3. 函数式 API 映射

| PyTorch | MLX |
|---|---|
| `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` |
| `F.relu(x)` | `mx.maximum(x, 0)` 或 `nn.relu(x)` |
| `F.log_softmax(x, dim=1)` | `mx.log_softmax(x, axis=1)` |
| `F.cross_entropy(pred, gold, ignore_index=pad)` | 需手动实现：用 `mx.losses.cross_entropy` + mask |
| `x.view(-1)` | `x.reshape(-1)` |

#### 4. 自动求导 — 最大不同

**PyTorch**（命令式）：
```python
optimizer.zero_grad()
loss = loss_fn(pred, gold)
loss.backward()
optimizer.step()
```

**MLX**（函数式）：
```python
def loss_fn(params, ...):
    pred = model(params, x)
    return mx.losses.cross_entropy(pred, gold, reduction='sum')

loss_and_grad = mx.value_and_grad(loss_fn)
loss, grads = loss_and_grad(model.trainable_parameters(), ...)
optimizer.update(model, grads)
mx.eval(model.state())  # 触发实际计算
```

---

### 逐文件改造要点

#### Modules.py — ScaledDotProductAttention

```python
# PyTorch 版本
attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
attn = attn.masked_fill(mask == 0, -1e9)
attn = self.dropout(F.softmax(attn, dim=-1))
output = torch.matmul(attn, v)

# MLX 版本
attn = (q / self.temperature) @ k.transpose(0, 2, 1, 3)
attn = mx.where(mask == 0, -1e9, attn)  # 注意：MLX 的 masked_fill
attn = self.dropout(mx.softmax(attn, axis=-1))
output = attn @ v
```

⚠️ **关键区别**：MLX 的 `transpose` 需要显式指定所有轴。

#### SubLayers.py — MultiHeadAttention + FFN

- `nn.Linear` → `mlx.nn.Linear`（一样）
- `nn.LayerNorm` → `mlx.nn.LayerNorm`（一样）
- `nn.Dropout` → `mlx.nn.Dropout`（一样）
- `q.transpose(1, 2)` → `mx.transpose(q, axes=(0,2,1,3))` 需要重写
- `q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)` → `q.reshape(sz_b, len_q, -1)`

#### Models.py — Transformer 主体

- `nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)` → `mlx.nn.Embedding(n_src_vocab, d_word_vec)`
- **无 `padding_idx`**：需要在 attention mask 中手动处理 padding
- `self.register_buffer('pos_table', ...)` → 直接用普通属性 + 在 `__init__` 中计算
- `nn.init.xavier_uniform_(p)` → `mlx.nn.init.xavier_uniform_(p)`（基本一样）
- 参数共享：`self.trg_word_prj.weight = self.decoder.trg_word_emb.weight` — MLX 也支持权重绑定

#### Optim.py — ScheduledOptim

可以直接复用大部分逻辑，只需将内部 `_optimizer`（`mlx.optimizers.Adam`）替换即可。
- `self._optimizer.step()` → `mx.eval(model.state())` 或直接等下一轮

#### Translator.py — Beam Search

- 没有 `register_buffer`，用普通属性即可
- `torch.LongTensor(...)` → `mx.array(..., dtype=mx.int64)`
- `.topk(k)` → `mx.topk(...)` 返回 (values, indices)

#### train.py — 训练流程

最大变化在训练循环：

```python
# MLX 训练循环范式
def train_step(model, optimizer, src, trg, gold):
    def loss_fn(params):
        pred = model(params, src, trg)
        loss = compute_loss(pred, gold, trg_pad_idx)
        return loss
    
    loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters(), ...)
    optimizer.update(model, grads)
    mx.eval(model.state(), optimizer.state)
    return loss
```

- **无需 `.to(device)`** — MLX 统一内存
- **无需 `model.eval()`/`model.train()`** — 用 `mx.nn.utils.eval(model)` 上下文管理器
- **数据加载**：MLX 可以用 `mx.utils.DataLoader` 或继续用 `torchtext` 数据然后转 `mx.array`

#### translate.py — 推理

- `torch.load` → `mx.load`（safetensors 格式）或用 `pickle` 继续存权重
- 加载模型时，需要将 PyTorch 权重转成 MLX 格式（注意 `nn.Embedding.weight` 形状兼容性）

---

### 推荐的渐进式迁移策略

| 阶段 | 内容 |
|---|---|
| **Phase 1** | 替换 Modules.py + SubLayers.py — 核心 attention + FFN |
| **Phase 2** | 替换 Layers.py + Models.py — 模型组装 |
| **Phase 3** | 重写 train.py 训练循环 — 函数式梯度 + 新优化器 API |
| **Phase 4** | 替换 Translator.py + translate.py — 推理兼容 |
| **Phase 5** | 数据加载适配 — 将 `torchtext` 数据转 `mx.array` 或改用 MLX 数据加载 |

### 已知陷阱

1. **`nn.Embedding` 无 `padding_idx`** — 需在 loss 计算和 attention mask 中手动忽略 padding
2. **MLX 的 `masked_fill`** — 使用 `mx.where(condition, value, tensor)` 替代
3. **随机种子** — `mx.random.seed(n)` 替代 `torch.manual_seed(n)`
4. **模型保存** — 推荐 safetensors 格式：`mx.save_safetensors(path, model_state)`
5. **`mx.eval()`** — 记得在 optimizer 更新后调用 `mx.eval(...)` 触发实际计算
6. **权重迁移** — 已有的 `.chkpt`（PyTorch 格式）需要脚本转换为 MLX 格式才能复用