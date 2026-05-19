Project: https://github.com/jadore801120/attention-is-all-you-need-pytorch

This project is too old and run with python 3.6, most of libraries in the project are deprecated or changed API.

This document explain how to run with python311

All help from chartgtp!!!

# History

## Session 1

That repo was written for very old versions: Python 3.6 + PyTorch 1.3 + torchtext/spacy APIs from ~2019.

You can still run it on modern Python (3.10–3.12), but you need to patch a few dependencies because:

- old `torchtext` APIs were removed
- old `spacy` model names changed
- some numpy/pickle behavior changed
- PyTorch internals evolved

The easiest path is:

- Python 3.10 or 3.11
- modern PyTorch
- replace old torchtext usage
- pin compatible package versions

Repository:

[attention-is-all-you-need-pytorch GitHub repo](https://github.com/jadore801120/attention-is-all-you-need-pytorch?utm_source=chatgpt.com)

---

# Recommended setup (works in 2026)

## 1. Create a fresh environment

### Linux / macOS

```
python3.11 -m venv venvsource venv/bin/activate
```

### Windows

```
py -3.11 -m venv venvvenv\Scripts\activate
```

---

# 2. Install modern dependencies

Do NOT use the repo's original `requirements.txt`.

Install this instead:

```
pip install torch torchvision torchaudiopip install spacy==3.7.5pip install torchtext==0.6.0pip install dill tqdm numpy
```

Then download language models:

```
python -m spacy download en_core_web_smpython -m spacy download de_core_news_sm
```

---

# 3. Fix the spaCy code

The repo uses old spaCy names:

OLD:

```
spacy.load('en')spacy.load('de')
```

Modern spaCy requires:

```
spacy.load("en_core_web_sm")spacy.load("de_core_news_sm")
```

Search the repo for:

```
spacy.load(
```

and replace accordingly.

---

# 4. Fix torchtext compatibility

This repo depends on the legacy torchtext API (`Field`, `BucketIterator`, etc.).

Modern torchtext removed many of these APIs.

You have 2 options:

## Option A (easiest)

Use:

```
pip install torchtext==0.6.0
```

This still contains the old APIs and works with Python 3.11.

---

## Option B (harder but future-proof)

Rewrite preprocessing using:

- HuggingFace datasets
- torch DataLoader
- custom tokenizer

Not worth it unless you're modernizing the project fully.

---

# 5. Run preprocessing

```
python preprocess.py \    -lang_src de \    -lang_trg en \    -share_vocab \    -save_data m30k_deen_shr.pkl
```

If you get a tokenizer error, edit tokenizer loading.

Find something like:

```
spacy_de = spacy.load('de')spacy_en = spacy.load('en')
```

Replace with:

```
spacy_de = spacy.load("de_core_news_sm")spacy_en = spacy.load("en_core_web_sm")
```

---

# 6. Train

```
python train.py \    -data_pkl m30k_deen_shr.pkl \    -log m30k_deen_shr \    -embs_share_weight \    -proj_share_weight \    -label_smoothing \    -output_dir output \    -b 256 \    -warmup 128000 \    -epoch 10
```

For small GPUs:

```
-b 32
```

or

```
-b 16
```

---

# Common modern errors

## `ModuleNotFoundError: torchtext.data`

Install legacy torchtext:

```
pip install torchtext==0.6.0
```

---

## `OSError: Can't find model 'en'`

Use:

```
python -m spacy download en_core_web_sm
```

and update `spacy.load()` calls.

---

## CUDA mismatch

Install PyTorch matching your CUDA version from:

[PyTorch official install page](https://pytorch.org/get-started/locally/?utm_source=chatgpt.com)

---

# Better alternative (recommended)

That repo is historically important, but outdated.

For learning Transformers on modern PyTorch, better maintained repos are:

- [Hugging Face Transformers](https://github.com/huggingface/transformers?utm_source=chatgpt.com)
- [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT?utm_source=chatgpt.com)
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html?utm_source=chatgpt.com)

They work cleanly with Python 3.11/3.12 and modern PyTorch.

---

# Minimal known-working versions

| Package   | Version      |
| --------- | ------------ |
| Python    | 3.10 or 3.11 |
| torch     | 2.3+         |
| torchtext | 0.6.0        |
| spacy     | 3.7.5        |
| numpy     | latest       |
| tqdm      | latest       |


## Session 2

``` log
python preprocess.py \ -lang_src de \ -lang_trg en \ -share_vocab \ -save_data m30k_deen_shr.pkl Namespace(lang_src='de', lang_trg='en', save_data='m30k_deen_shr.pkl', data_src=None, data_trg=None, max_len=100, min_word_count=3, keep_case=False, share_vocab=True) Traceback (most recent call last): File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 335, in <module> main_wo_bpe() File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 270, in main_wo_bpe src_lang_model = spacy.load(opt.lang_src) ^^^^^^^^^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/spacy/__init__.py", line 51, in load return util.load_model( ^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/spacy/util.py", line 471, in load_model raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name])) # type: ignore[index] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ OSError: [E941] Can't find model 'de'. It looks like you're trying to load a model from a shortcut, which is obsolete as of spaCy v3.0. To load the model, use its full name instead: nlp = spacy.load("de_core_news_sm") For more details on the available models, see the models directory: https://spacy.io/models and if you want to create a blank model, use spacy.blank: nlp = spacy.blank("de")
```

This is the expected spaCy v3 breakage.

The repo still uses old spaCy v2 shortcuts:

```
spacy.load('de')spacy.load('en')
```

Modern spaCy requires full model names.

---

# Fix

Open:

```
preprocess.py
```

Find this section (around line ~270):

```
src_lang_model = spacy.load(opt.lang_src)trg_lang_model = spacy.load(opt.lang_trg)
```

Replace with:

```
if opt.lang_src == "de":    src_lang_model = spacy.load("de_core_news_sm")elif opt.lang_src == "en":    src_lang_model = spacy.load("en_core_web_sm")else:    src_lang_model = spacy.blank(opt.lang_src)if opt.lang_trg == "de":    trg_lang_model = spacy.load("de_core_news_sm")elif opt.lang_trg == "en":    trg_lang_model = spacy.load("en_core_web_sm")else:    trg_lang_model = spacy.blank(opt.lang_trg)
```

---

# Then install the models

Run:

```
python -m spacy download de_core_news_smpython -m spacy download en_core_web_sm
```

---

# Then rerun

```
python preprocess.py \    -lang_src de \    -lang_trg en \    -share_vocab \    -save_data m30k_deen_shr.pkl
```

---

# Faster minimal fix

If you only care about German→English, the shortest patch is:

```
src_lang_model = spacy.load("de_core_news_sm")trg_lang_model = spacy.load("en_core_web_sm")
```

---

# Another issue you may hit next

Modern `torchtext` sometimes breaks pickle serialization.

If you later get errors involving:

- `Field`
- `Dataset`
- pickle
- `dill`

install:

```
pip install dill
```

or downgrade torchtext further:

```
pip install torchtext==0.6.0
```

---

# Verify models installed

Run:

```
python -m spacy validate
```

You should see something like:

```
✔ Loaded compatibility table================= Installed pipeline packages =================NAME               SPACY            VERSIONde_core_news_sm    >=3.7.0,<3.8.0   3.7.0en_core_web_sm     >=3.7.0,<3.8.0   3.7.1
```

After that preprocessing should proceed normally.


## Session 3

``` log
python preprocess.py \ -lang_src de_core_news_sm \ -lang_trg en_core_web_sm \ -share_vocab \ -save_data m30k_deen_shr.pkl Namespace(lang_src='de_core_news_sm', lang_trg='en_core_web_sm', save_data='m30k_deen_shr.pkl', data_src=None, data_trg=None, max_len=100, min_word_count=3, keep_case=False, share_vocab=True) Traceback (most recent call last): File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 335, in <module> main_wo_bpe() File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 299, in main_wo_bpe train, val, test = torchtext.datasets.Multi30k.splits( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 113, in splits return super(Multi30k, cls).splits( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 65, in splits train_data = None if train is None else cls( ^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 34, in __init__ with io.open(src_path, mode='r', encoding='utf-8') as src_file, \ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ FileNotFoundError: [Errno 2] No such file or directory: '.data/multi30k/train.de_core_news_sm'
```

Now the problem is different: you changed the dataset language code too.

`torchtext.datasets.Multi30k` expects dataset extensions:

- `de`
- `en`

NOT spaCy model names.

So this command is wrong:

```
-lang_src de_core_news_sm \-lang_trg en_core_web_sm
```

because torchtext tries to load files like:

```
train.de_core_news_sm
```

which do not exist.

---

# Correct approach

Keep dataset language codes:

```
-lang_src de \-lang_trg en
```

BUT modify the spaCy loading code inside `preprocess.py`.

---

# Do this

## 1. Edit `preprocess.py`

Find:

```
src_lang_model = spacy.load(opt.lang_src)trg_lang_model = spacy.load(opt.lang_trg)
```

Replace with:

```
src_lang_model = spacy.load("de_core_news_sm")trg_lang_model = spacy.load("en_core_web_sm")
```

---

# 2. Run the ORIGINAL command

```
python preprocess.py \    -lang_src de \    -lang_trg en \    -share_vocab \    -save_data m30k_deen_shr.pkl
```

---

# Why this happens

The script uses:

```
exts = ('.' + opt.lang_src, '.' + opt.lang_trg)
```

for dataset filenames.

So:

- `de` → `train.de`
- `en` → `train.en`

Changing them to spaCy model names breaks dataset loading.

---

# You may hit another Multi30k issue next

Modern torchtext sometimes fails downloading Multi30k because old URLs died.

If that happens, the cleanest fix is:

```
mkdir -p .data/multi30k
```

Then manually download:

- train.de
- train.en
- val.de
- val.en
- test.de
- test.en

from:

[Multi30k dataset GitHub mirror](https://github.com/multi30k/dataset?utm_source=chatgpt.com)

Place them into:

```
.data/multi30k/
```

Then preprocessing will work.

## Session 4

```log
python preprocess.py \ -lang_src de \ -lang_trg en \ -share_vocab \ -save_data m30k_deen_shr.pkl Namespace(lang_src='de', lang_trg='en', save_data='m30k_deen_shr.pkl', data_src=None, data_trg=None, max_len=100, min_word_count=3, keep_case=False, share_vocab=True) Traceback (most recent call last): File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 337, in <module> main_wo_bpe() File "/Users/henry/git/AI-Projects/attention-is-all-you-need-pytorch/preprocess.py", line 301, in main_wo_bpe train, val, test = torchtext.datasets.Multi30k.splits( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 113, in splits return super(Multi30k, cls).splits( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 65, in splits train_data = None if train is None else cls( ^^^^ File "/opt/homebrew/Caskroom/miniconda/base/envs/transformer_py311/lib/python3.11/site-packages/torchtext/datasets/translation.py", line 34, in __init__ with io.open(src_path, mode='r', encoding='utf-8') as src_file, \ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ FileNotFoundError: [Errno 2] No such file or directory: '.data/multi30k/train.de'
```

Good — now spaCy is fixed.  
This error means the dataset was never downloaded.

Old `torchtext` download URLs for Multi30k are partially broken now, so the automatic download often fails silently on modern systems.

You need to manually place the dataset files.

---

# Fix Multi30k dataset manually

## 1. Create dataset directory

From the repo root:

```
mkdir -p .data/multi30k
```

---

# 2. Download the files

Get these 6 files:

```
train.detrain.enval.deval.entest.detest.en
```

from:

[Multi30k dataset mirror](https://github.com/multi30k/dataset?utm_source=chatgpt.com)

A direct raw source that still works:

[Multi30k data task1 raw files](https://github.com/multi30k/dataset/tree/master/data/task1/raw?utm_source=chatgpt.com)

---

# 3. Put them here

Your structure should become:

```
attention-is-all-you-need-pytorch/├── preprocess.py├── train.py└── .data/    └── multi30k/        ├── train.de        ├── train.en        ├── val.de        ├── val.en        ├── test.de        └── test.en
```

---

# 4. Run preprocess again

```
python preprocess.py \    -lang_src de \    -lang_trg en \    -share_vocab \    -save_data m30k_deen_shr.pkl
```

---

# If GitHub downloads HTML pages instead of text

Use terminal downloads instead.

Example:

```
cd .data/multi30kcurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.decurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.encurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.decurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.encurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test.decurl -O https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test.en
```

---

# After preprocessing succeeds

Run training:

```
python train.py \    -data_pkl m30k_deen_shr.pkl \    -log transformer \    -output_dir output \    -b 32 \    -epoch 10
```

Use:

- `-b 16` for Apple Silicon 8GB
- `-b 32` for 16GB+
- `-no_cuda` if CUDA detection acts weird on macOS

Example:

```
python train.py \    -data_pkl m30k_deen_shr.pkl \    -log transformer \    -output_dir output \    -b 16 \    -epoch 10 \    -no_cuda
```

## Session 5

```log
❯ python train.py \ -data_pkl m30k_deen_shr.pkl \ -log transformer \ -output_dir output \ -b 32 \ -epoch 10 usage: train.py [-h] [-data_pkl DATA_PKL] [-train_path TRAIN_PATH] [-val_path VAL_PATH] [-epoch EPOCH] [-b BATCH_SIZE] [-d_model D_MODEL] [-d_inner_hid D_INNER_HID] [-d_k D_K] [-d_v D_V] [-n_head N_HEAD] [-n_layers N_LAYERS] [-warmup N_WARMUP_STEPS] [-lr_mul LR_MUL] [-seed SEED] [-dropout DROPOUT] [-embs_share_weight] [-proj_share_weight] [-scale_emb_or_prj SCALE_EMB_OR_PRJ] [-output_dir OUTPUT_DIR] [-use_tb] [-save_mode {all,best}] [-no_cuda] [-label_smoothing] train.py: error: unrecognized arguments: -log transformer
```

That means the repo version you cloned is slightly different from the original examples online.

Your `train.py` does not support:

```
-log transformer
```

Just remove it.

Run:

```
python train.py \    -data_pkl m30k_deen_shr.pkl \    -output_dir output \    -b 32 \    -epoch 10
```

---

# For Apple Silicon / macOS

If CUDA-related issues appear, use:

```
python train.py \    -data_pkl m30k_deen_shr.pkl \    -output_dir output \    -b 16 \    -epoch 10 \    -no_cuda
```

---

# You may hit this next

Older code often contains:

```
tensor.cuda()
```

even when CUDA is disabled.

If training crashes with CUDA/device errors on Mac, patch `train.py`.

Search for:

```
.cuda()
```

and replace with:

```
.to(device)
```

where `device` is:

```
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

for Apple Silicon.

---

# Better Apple Silicon patch (recommended)

Near the top of `train.py`, after imports, add:

```
import torchdevice = torch.device(    "mps" if torch.backends.mps.is_available()    else "cuda" if torch.cuda.is_available()    else "cpu")
```

Then replace all:

```
.cuda()
```

with:

```
.to(device)
```

This modernizes the repo for:

- Apple M-series GPUs
- CUDA GPUs
- CPU fallback

---

# If training starts successfully

You should eventually see logs like:

```
[Info] Training performance will be written to file.[ Epoch 0 ]
```

and loss/perplexity updates.

## Session 6

``` log
[ Epoch 1 ] - (Training) ppl: 108.58289, accuracy: 25.647 %, lr: 0.00032, elapse: 4.592 min - (Validation) ppl: 86.92886, accuracy: 27.458 %, lr: 0.00032, elapse: 0.046 min - [Info] The checkpoint file has been updated. [ Epoch 2 ] - (Training) ppl: 63.96428, accuracy: 30.835 %, lr: 0.00048, elapse: 4.596 min - (Validation) ppl: 49.75328, accuracy: 32.985 %, lr: 0.00048, elapse: 0.046 min - [Info] The checkpoint file has been updated. [ Epoch 3 ] - (Training) ppl: 46.49185, accuracy: 33.331 %, lr: 0.00063, elapse: 4.603 min - (Validation) ppl: 37.98506, accuracy: 34.730 %, lr: 0.00063, elapse: 0.046 min - [Info] The checkpoint file has been updated. [ Epoch 4 ] - (Training) ppl: 42.08174, accuracy: 34.025 %, lr: 0.00079, elapse: 4.591 min - (Validation) ppl: 41.34377, accuracy: 33.954 %, lr: 0.00079, elapse: 0.045 min [ Epoch 5 ] - (Training) ppl: 74.87738, accuracy: 27.032 %, lr: 0.00095, elapse: 4.602 min - (Validation) ppl: 834.95421, accuracy: 11.988 %, lr: 0.00095, elapse: 0.047 min [ Epoch 6 ] - (Training) ppl: 173.58378, accuracy: 14.415 %, lr: 0.00111, elapse: 4.603 min - (Validation) ppl: 3014.37072, accuracy: 11.988 %, lr: 0.00111, elapse: 0.043 min [ Epoch 7 ] - (Training) ppl: 168.29413, accuracy: 14.759 %, lr: 0.00127, elapse: 4.620 min - (Validation) ppl: 2950.48175, accuracy: 11.988 %, lr: 0.00127, elapse: 0.046 min [ Epoch 8 ] - (Training) ppl: 165.60762, accuracy: 14.996 %, lr: 0.00138, elapse: 4.602 min - (Validation) ppl: 2115.13691, accuracy: 11.988 %, lr: 0.00138, elapse: 0.047 min [ Epoch 9 ] - (Training) ppl: 163.57107, accuracy: 15.176 %, lr: 0.00131, elapse: 4.614 min - (Validation) ppl: 2254.47724, accuracy: 11.988 %, lr: 0.00131, elapse: 0.046 min 为什么训练效果反而在下降
```

这是典型的：

```
learning rate 爆了
```

不是数据问题。

不是 MLX 问题。

也不是 Transformer 坏了。

你这个日志非常经典：

```
epoch 1~4 正常下降epoch 5 开始突然爆炸
```

看这里：

|epoch|lr|
|---|---|
|1|0.00032|
|2|0.00048|
|3|0.00063|
|4|0.00079|
|5|0.00095 ← 开始炸|
|6|0.00111|
|7|0.00127|

---

# 直接原因

你还在：

```
warmup阶段
```

但：

```
batch 太小
```

导致：

```
学习率增长太快
```

---

# 这是原论文的坑

Attention Is All You Need 原始配置：

|参数|原论文|
|---|---|
|batch|2048|
|warmup|4000|
|GPU|8×P100|

而你：

```
batch=32
```

甚至可能更小。

但：

```
warmup_steps=4000
```

还保持原值。

于是：

```
单位 token 的 lr 增长过猛
```

---

# 你的 train.py 其实已经警告了

你代码里：

```
if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:    print('[Warning] The warmup steps may be not enough.')
```

它已经提醒你了。

---

# 为什么 epoch 4 后突然爆炸

因为：

```
warmup lr 越来越高
```

直到：

```
Adam 更新不稳定
```

然后：

```
attention logits 爆炸↓softmax overflow↓loss 爆炸↓模型崩掉
```

---

# 典型症状

你现在：

|指标|表现|
|---|---|
|ppl|从 40 → 3000|
|accuracy|掉到 11%|
|validation|完全崩|

这是：

```
optimizer divergence
```

标准现象。

---

# 最简单修复（立刻有效）

---

# 方法 1（最推荐）

把 warmup 提高。

例如：

```
-warmup 20000
```

甚至：

```
-warmup 40000
```

---

# 原因

小 batch 时：

```
需要更长 warmup
```

---

# 方法 2（非常有效）

降低 lr multiplier。

现在默认：

```
-lr_mul 2.0
```

改成：

```
-lr_mul 0.5
```

甚至：

```
-lr_mul 0.2
```

---

# 方法 3（最稳）

直接 gradient clipping。

在：

```
loss.backward()
```

后面加入：

```
torch.nn.utils.clip_grad_norm_(    model.parameters(),    1.0)
```

---

# 推荐完整修复

你现在直接：

```
python train.py \  -data_pkl m30k_deen_shr.pkl \  -output_dir output \  -b 32 \  -epoch 30 \  -warmup 20000 \  -lr_mul 0.5 \  -label_smoothing
```

---

# 你会看到：

```
ppl 持续下降
```

而不是：

```
epoch 5 爆炸
```

---

# 更深层原因（Transformer 老问题）

原始 Transformer：

```
Post-LN
```

结构：

```
x + attention↓LayerNorm
```

非常不稳定。

现代模型：

```
Pre-LN
```

即：

```
LayerNorm↓attention↓residual
```

稳定很多。

---

# 你的 repo 是：

```
2019 老 Transformer
```

所以：

```
更容易训练爆炸
```

---

# 真正现代解决方案

如果你继续现代化：

应该：

---

# 1. Pre-LN

替换：

```
x = self.norm(x + attn)
```

为：

```
x = x + attn(self.norm(x))
```

---

# 2. RMSNorm

比 LayerNorm 更稳定。

---

# 3. bf16

Apple Silicon 很适合。

---

# 4. Flash Attention

attention 数值更稳定。

---

# 5. cosine lr schedule

比 old warmup 更稳定。

---

# 6. AdamW

不是 Adam。

---

# 7. gradient clipping

现代训练几乎必备。

---

# 你现在的日志其实说明：

```
模型是能学到东西的
```

因为：

|epoch|ppl|
|---|---|
|1|108|
|2|63|
|3|46|
|4|42|

说明：

- tokenizer 没问题
- 数据没问题
- attention 没问题
- pipeline 没问题

只是：

```
学习率策略炸了
```

这是好消息。

因为这比：

- 数据错
- mask错
- shift错
- decoder错

容易修。

Gradient clipping（梯度裁剪）是：

```
限制梯度不要过大
```

防止训练时：

```
参数一步更新太猛
```

导致模型崩掉。

---

# 为什么 Transformer 特别需要它

Transformer 很容易出现：

```
gradient explosion（梯度爆炸）
```

尤其：

- learning rate 太大
- warmup 不合理
- batch 太小
- 深层网络
- attention logits 过大

时。

你刚才那个：

```
epoch 5 突然 ppl 爆炸
```

就是典型 gradient explosion。

---

# 不 clipping 会发生什么

例如某层梯度：

```
grad = 5837
```

Adam update：

```
weight -= lr * grad
```

参数会：

```
瞬间飞掉
```

然后：

- logits 极大
- softmax overflow
- loss nan
- accuracy 崩

---

# clipping 做什么

例如：

```
clip_grad_norm_(..., 1.0)
```

意思：

```
如果整体梯度 norm > 1就缩小它
```

---

# 举例

原梯度：

```
||g|| = 12
```

clip 到：

```
1.0
```

会变成：

```
g = g / 12
```

即：

```
所有梯度同比缩小
```

方向不变。

---

# 数学上

如果：

∥g∥>c\|g\| > c∥g∥>c

则：

g←g⋅c∥g∥g \leftarrow g \cdot \frac{c}{\|g\|}g←g⋅∥g∥c​

其中：

- ggg = gradient
- ccc = clipping threshold

---

# 最常见 clipping

---

# 1. norm clipping（最常用）

PyTorch：

```
torch.nn.utils.clip_grad_norm_(    model.parameters(),    1.0)
```

意思：

```
整体梯度 norm 最大 1
```

这是 Transformer 最常用。

---

# 2. value clipping

例如：

```
clip_grad_value_(    model.parameters(),    1.0)
```

直接：

```
每个元素限制在 [-1,1]
```

较少用。

---

# 为什么 norm clipping 更好

因为：

```
保留梯度方向
```

只缩放大小。

---

# Transformer 里几乎是标配

现代模型基本都会：

✅ gradient clipping  
✅ warmup  
✅ AdamW  
✅ Pre-LN  
✅ bf16/fp16 stabilization

一起用。

---

# 你现在应该加在哪里

在：

```
loss.backward()
```

后面。

---

# 你的 train.py

改：

```
loss.backward()optimizer.step_and_update_lr()
```

为：

```
loss.backward()torch.nn.utils.clip_grad_norm_(    model.parameters(),    1.0)optimizer.step_and_update_lr()
```

---

# 为什么 1.0 最常见

经验上：

|threshold|效果|
|---|---|
|0.1|太小，学不动|
|1.0|最常见|
|5.0|偏宽松|
|10|基本没 clip|

Transformer 通常：

```
0.5 ~ 1.0
```

---

# 你这个项目最推荐

```
clip_grad_norm_(..., 1.0)
```

配合：

```
-lr_mul 0.5-warmup 20000
```

基本就稳了。

---

# 为什么现代 LLM 更稳定

因为它们已经：

|技术|作用|
|---|---|
|Pre-LN|稳定梯度|
|RMSNorm|稳定scale|
|RoPE|长文本稳定|
|AdamW|更稳定优化|
|clipping|防爆|
|FlashAttention|数值稳定|
|bf16|更稳更快|

---

# 你的 repo 属于：

```
第一代 Transformer
```

所以：

```
更容易炸
```

加 clipping 会明显改善。

没有归一化和残差很容易爆是吧

对。

Transformer 如果：

```
没有 Residual没有 Normalization
```

会：

```
非常容易训练崩掉
```

尤其 attention 网络。

---

# 为什么会爆

深层网络本质上：

```
连续函数复合
```

例如：

f8(f7(f6(...f1(x))))f_8(f_7(f_6(...f_1(x))))f8​(f7​(f6​(...f1​(x))))

每一层都会：

- 放大
- 缩小
- 扭曲

数值。

---

# 没有 residual 时

梯度传播：

```
只能穿过所有层
```

会出现：

---

# 1. 梯度消失

0.950≈0.0050.9^{50} \approx 0.0050.950≈0.005

越来越小。

学不动。

---

# 2. 梯度爆炸

1.150≈1171.1^{50} \approx 1171.150≈117

越来越大。

直接炸。

---

# Residual 的作用

Transformer：

```
x = x + attention(x)
```

这个：

```
+ x
```

极其重要。

---

# 为什么

因为它提供：

```
identity path
```

即：

```
至少能原样传递
```

梯度。

---

# 数学上非常关键

没有 residual：

xl+1=f(xl)x_{l+1}=f(x_l)xl+1​=f(xl​)

有 residual：

xl+1=xl+f(xl)x_{l+1}=x_l+f(x_l)xl+1​=xl​+f(xl​)

---

# 后者稳定得多

因为：

∂xl+1∂xl=1+∂f∂xl\frac{\partial x_{l+1}}{\partial x_l}=1+\frac{\partial f}{\partial x_l}∂xl​∂xl+1​​=1+∂xl​∂f​

这里有个：

```
1
```

梯度永远不会完全消失。

这就是：

- ResNet
- Transformer
- Llama

能做很深的原因。

---

# Normalization 为什么也重要

attention 输出可能：

```
scale 飘掉
```

例如：

```
layer1 variance = 1layer5 variance = 200layer20 variance = 50000
```

最后：

```
softmax overflow
```

直接炸。

---

# LayerNorm / RMSNorm 做什么

本质：

```
把 hidden state 拉回稳定范围
```

---

# LayerNorm

近似：

x←x−μσx \leftarrow \frac{x-\mu}{\sigma}x←σx−μ​

---

# RMSNorm

现代 LLM 更喜欢：

x←x1n∑xi2x \leftarrow \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2}}x←n1​∑xi2​​x​

更简单。

更稳定。

更快。

---

# 为什么 Attention 特别容易爆

因为：

attention 里有：

QKTQK^TQKT

如果：

```
Q/K scale 变大
```

那么：

```
softmax 会极端化
```

例如：

```
[1000, -300, 500]
```

softmax：

```
≈ [1, 0, 0]
```

梯度直接坏掉。

---

# 所以 Transformer 稳定训练依赖：

|组件|作用|
|---|---|
|Residual|保梯度|
|LayerNorm/RMSNorm|保scale|
|Warmup|防 early explosion|
|Gradient Clipping|防瞬时爆炸|
|Attention scaling|防 softmax overflow|

---

# 为什么论文里有：

QKTdk\frac{QK^T}{\sqrt{d_k}}dk​​QKT​

就是为了：

```
降低 attention logits scale
```

否则：

```
d_k=512
```

时：

```
QK^T 非常容易巨大
```

---

# 现代 Transformer 更稳定

现代模型已经：

|老 Transformer|现代 LLM|
|---|---|
|Post-LN|Pre-LN|
|LayerNorm|RMSNorm|
|无 clipping|clipping|
|naive attention|flash attention|
|fp32|bf16|
|Adam|AdamW|

---

# 你那个 repo 属于：

```
第一代 Transformer
```

所以：

```
数值稳定性比较差
```

是正常的。


## Session 7

```log
🔥 [ Epoch 50 ] - (Training) ppl: 22.70679, accuracy: 51.811 %, lr: 0.00015, elapse: 0.799 min - (Validation) ppl: 15.06064, accuracy: 48.338 %, lr: 0.00015, elapse: 0.006 min 🔥 [ Epoch 51 ] - (Training) ppl: 22.30286, accuracy: 52.036 %, lr: 0.00015, elapse: 0.787 min - (Validation) ppl: 15.09177, accuracy: 48.082 %, lr: 0.00015, elapse: 0.006 min 🔥 [ Epoch 52 ] - (Training) ppl: 21.97096, accuracy: 52.326 %, lr: 0.00015, elapse: 0.781 min - (Validation) ppl: 15.15825, accuracy: 48.241 %, lr: 0.00015, elapse: 0.006 min 🔥 [ Epoch 53 ] - (Training) ppl: 21.59712, accuracy: 52.587 %, lr: 0.00015, elapse: 0.784 min - (Validation) ppl: 15.03274, accuracy: 48.227 %, lr: 0.00015, elapse: 0.006 min 运行多个epoch后，validation的accuracy上的非常慢，一直超不过50%
```


这是正常现象。

你现在已经进入：

```
Transformer 小数据集瓶颈区
```

不是代码坏了。

不是训练失败。

而是：

```
Multi30k 太小
```

- 你的模型已经快吃满数据了。

---

# 其实你现在效果已经不错了

你现在：

|指标|数值|
|---|---|
|train acc|52%|
|valid acc|48%|
|ppl|15|

对于：

```
从零训练的小 Transformer
```

已经是正常水平。

---

# 为什么 validation 卡住

因为：

```
token-level accuracy
```

本来就很难高。

尤其机器翻译。

---

# 例如：

真实：

```
A man is playing guitar.
```

预测：

```
A person is playing guitar.
```

语义几乎一样。

但：

```
accuracy 会认为错
```

---

# token accuracy 本身就不是翻译核心指标

真正翻译指标是：

|指标|用途|
|---|---|
|BLEU|翻译质量|
|chrF|字符级|
|COMET|现代语义|
|perplexity|language modeling|

不是：

```
token accuracy
```

---

# 为什么 50% 很常见

翻译是：

```
多解问题
```

例如：

|Reference|Prediction|
|---|---|
|a man is walking|a person walks|
|children are running|kids run|

token 不一样。

但翻译没问题。

---

# 更深层原因

Multi30k：

```
只有 29k 样本
```

非常小。

现代 Transformer：

```
几百万参数
```

很容易：

```
学完 dataset
```

然后：

- validation 不再涨
- 开始 noise fitting
- generalization 到顶

---

# 你现在的迹象

看：

|epoch|valid ppl|
|---|---|
|50|15.06|
|51|15.09|
|52|15.15|
|53|15.03|

已经：

```
plateau
```

了。

---

# 为什么 train 还能涨

因为：

```
模型开始记忆训练集
```

而不是：

- 学习语言规律
- 提升泛化

---

# 这是 overfitting 前期

典型表现：

|train|valid|
|---|---|
|持续变好|不再明显变好|

---

# 为什么 Transformer 特别容易这样

因为：

- attention capacity 很强
- Multi30k 很小
- vocab 不大
- 句子很短

---

# 真正限制你的东西

不是：

- optimizer
- GPU
- MLX

而是：

```
数据规模
```

---

# 真正提升方法

---

# 1. 更大数据集（最重要）

例如：

|数据集|规模|
|---|---|
|Multi30k|29k|
|IWSLT14|160k|
|WMT14|4.5M|

差了：

```
两个数量级
```

---

# 2. label smoothing

非常重要。

你代码支持：

```
-label_smoothing
```

它会明显提升：

- BLEU
- 泛化

虽然 accuracy 不一定涨。

---

# 3. dropout

现在：

```
dropout=0.1
```

可以试：

```
-dropout 0.3
```

---

# 4. 更小模型

Multi30k 不适合：

```
6-layer transformer
```

太大了。

---

# 推荐：

```
-n_layers 3-d_model 256-n_head 4
```

会更容易泛化。

---

# 5. BLEU evaluation（重点）

你现在最该做的是：

```
不要看 accuracy
```

看：

```
BLEU
```

---

# 因为翻译任务：

```
BLEU 才是真指标
```

---

# 例如

accuracy：

```
48%
```

但 BLEU：

```
28
```

其实已经不错。

---

# 现代 LLM 甚至不用 accuracy

因为：

```
token accuracy 不反映生成质量
```

---

# 你现在已经说明：

✅ model 在学习  
✅ attention 正常  
✅ optimizer 正常  
✅ masking 正常  
✅ positional encoding 正常  
✅ data pipeline 正常

这是非常关键的。

---

# 你现在真正应该升级的是：

---

# 1. BLEU metric

不要只看 accuracy。

---

# 2. inference/generate.py

真正翻译句子。

---

# 3. beam search

greedy decoding 很弱。

---

# 4. bigger dataset

这是提升最大来源。

---

# 5. SentencePiece tokenizer

会比老 torchtext tokenizer 好很多。

---

# 其实你现在已经：

```
成功训练了一个真正的 Transformer
```


## Session 8

昨晚把用mac的cpu训练换成了用gpu训练，修改了下面这行代码。发现cpu训练一个epoch要5分钟，gpu只要40-50秒，提升了5倍性能，当然修改大了batch size 到256， cpu之前用的是32，但是cpu的batch size改大了也快不了。

```python
device = torch.device("cuda" if opt.cuda else ("mps" if torch.backends.mps.is_available() else "cpu"))
```
