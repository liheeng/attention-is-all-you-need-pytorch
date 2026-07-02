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