# TactileACT - Visuo-Tactile Pretraining for Robotic Manipulation

This repository implements Action Chunking Transformer (ACT) with visuo-tactile pretraining for robotic manipulation tasks.

Paper: https://arxiv.org/abs/2403.11898

## 环境配置 (Environment Setup)

### 1. 创建并激活conda环境
```bash
conda create -n TactileACT python=3.8
conda activate TactileACT
```

### 2. 安装依赖包
```bash
pip install torch torchvision
pip install opencv-python matplotlib einops h5py ipython tqdm
pip install pyyaml pexpect packaging
```

### 3. 安装DETR模块
```bash
cd detr && pip install -e . && cd ..
```

## 数据准备

确保你的数据文件组织如下：
```
data/data_dir/
├── meta_data.json          # 数据集元信息
├── data/
│   ├── episode_0.hdf5      # HDF5格式的训练数据
│   ├── episode_1.hdf5
│   └── ...
└── clip_models/            # （可选）预训练的CLIP模型
    ├── vision_encoder.pth
    └── gelsight_encoder.pth
```

## CLIP预训练 (可选)

如果需要训练视觉-触觉编码器，使用CLIP风格的对比学习：

```bash
python clip_pretraining.py --data_dir ./data/data_dir/data --num_episodes 100 --batch_size 32 --num_epochs 100 --lr 0.0001
```

**主要参数说明：**
- `--data_dir`: HDF5数据文件所在目录
- `--num_episodes`: 训练使用的episode数量
- `--batch_size`: 批大小
- `--num_epochs`: 训练轮数
- `--lr`: 学习率
- `--save_path`: 模型保存路径（默认：`./data/clip_models/`）

训练完成后，会在指定路径生成：
- `vision_encoder.pth` - 视觉编码器权重
- `gelsight_encoder.pth` - 触觉编码器权重

## ACT模型训练

### 方式一：不使用预训练编码器

适用于没有进行CLIP预训练或想从头训练整个模型的情况。

#### 配置文件 (base_config.json)
```json
{
    "eval": false,
    "save_dir": "./data/data_dir",
    "name": "training_no_pretrain",
    "policy_class": "ACT",
    "batch_size": 2,
    "seed": 0,
    "num_epochs": 2000,
    "lr": 0.00001,
    "kl_weight": 10.0,
    "chunk_size": 30,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "temporal_agg": true,
    "z_dimension": 32,
    "gpu": 0,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "backbone": "resnet18",
    "position_embedding": "sine",
    "enc_layers": 4,
    "dec_layers": 7,
    "dropout": 0.025,
    "nheads": 8,
    "pre_norm": false,
    "masks": false,
    "gelsight_backbone_path": "none",
    "vision_backbone_path": "none"
}
```

#### 训练命令
```bash
python imitate_episodes.py --config base_config.json
```

### 方式二：使用预训练编码器

使用CLIP预训练的编码器，通常能获得更好的性能。

#### 配置文件修改
将 `base_config.json` 中的以下参数修改为：
```json
{
    "name": "training_with_pretrain",
    "backbone": "clip_backbone",
    "gelsight_backbone_path": "./data/clip_models/gelsight_encoder.pth",
    "vision_backbone_path": "./data/clip_models/vision_encoder.pth"
}
```

#### 训练命令
```bash
python imitate_episodes.py --config base_config.json \
    --name training_with_pretrain \
    --backbone clip_backbone \
    --gelsight_backbone_path ./data/clip_models/gelsight_encoder.pth \
    --vision_backbone_path ./data/clip_models/vision_encoder.pth
```

## 主要配置参数说明

### 基础参数
- `save_dir`: 数据目录，程序会在此目录下查找 `data/` 子文件夹和 `meta_data.json`
- `name`: 训练名称，会在 `save_dir` 下创建对应目录保存模型和日志
- `gpu`: 使用的GPU编号（-1表示使用CPU）
- `num_epochs`: 训练轮数
- `batch_size`: 批大小

### ACT模型参数
- `chunk_size`: 动作块大小，一次预测多少步未来动作（默认30）
- `kl_weight`: KL散度损失权重，用于CVAE训练（默认10.0）
- `z_dimension`: CVAE潜变量维度（默认32）
- `hidden_dim`: Transformer隐藏层维度（默认512）
- `dim_feedforward`: 前馈网络维度（默认3200）

### Transformer参数
- `enc_layers`: 编码器层数（默认4）
- `dec_layers`: 解码器层数（默认7）
- `nheads`: 注意力头数（默认8）
- `dropout`: Dropout比例（默认0.025）

### 学习率参数
- `lr`: 主学习率（默认1e-5）
- `lr_backbone`: 视觉backbone学习率（默认1e-5）
- `weight_decay`: 权重衰减（默认1e-4）

### 预训练相关
- `backbone`: 使用的backbone类型
  - `"resnet18"`: 标准ResNet18（不使用预训练）
  - `"clip_backbone"`: 使用CLIP预训练的编码器
- `gelsight_backbone_path`: 触觉编码器权重路径（使用clip_backbone时必需）
- `vision_backbone_path`: 视觉编码器权重路径（使用clip_backbone时必需）

## 训练输出

训练过程中会在 `save_dir/name/` 目录下生成：
- `policy_best.ckpt` - 验证集最佳模型
- `policy_last.ckpt` - 最后一轮模型
- `policy_epoch_X_seed_Y.ckpt` - 定期保存的检查点
- `args.json` - 训练配置
- `dataset_stats.pkl` - 数据归一化统计
- `train_history.pkl` / `validation_history.pkl` - 训练历史
- `visualizations/` - 训练可视化图像
- `train_val_*.png` - 损失曲线图

## 执行训练好的策略

```bash
python robot_operation.py --checkpoint ./data/data_dir/training_name/policy_best.ckpt
```

## 联系方式

如有问题或需要帮助，请联系：aigeorge@andrew.cmu.edu

