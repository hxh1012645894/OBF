# CLIP + LoRA + CoOp 集成说明

本项目已将CLIP模型与LoRA微调、CoOp(Context Optimization)技术集成到半监督学习框架中。

## 目录结构

```
├── semilearn/nets/clip_lora_coop/
│   ├── clip_lora_coop.py      # 核心模型实现
│   ├── prompt.json            # 类别描述文件
│   └── __init__.py            # 模型导出
├── config/clip_lora_coop/
│   ├── clip_lora_coop_full.yaml      # 完整模式：LoRA + CoOp
│   ├── clip_lora_coop_large.yaml     # 大型LoRA (r=16)
│   ├── clip_lora_only.yaml           # 仅LoRA模式
│   ├── clip_coop_only.yaml           # 仅CoOp模式
│   ├── clip_zero_shot.yaml           # 零样本CLIP
│   ├── clip_linear_probe.yaml        # 线性探测模式
│   └── clip_full_finetune.yaml       # 全参数微调
├── train_clip_lora_coop.py    # 专用训练脚本
├── train_coop.py              # CoOp训练脚本
└── eval_zeroshot_clip.py      # 零样本评估脚本
```

## 支持的训练模式

| 模式 | LoRA | CoOp | 视觉骨干 | 可训练参数 |
|------|------|------|----------|------------|
| Full | ✓ | ✓ | Frozen | LoRA适配器 + Context Tokens |
| LoRA-only | ✓ | ✗ | Frozen | LoRA适配器 |
| CoOp-only | ✗ | ✓ | Frozen | Context Tokens |
| Zero-shot | ✗ | ✗ | Frozen | 无（纯推理） |
| Linear Probe | ✗ | ✗ | Frozen | 线性分类器 |
| Full Fine-tuning | ✗ | ✓ | Unfrozen | 全部参数 |

## 快速开始

### 1. 使用专用训练脚本

```bash
# 完整模式训练
python train_clip_lora_coop.py --mode full --data_dir ./data --num_classes 9

# 仅LoRA模式
python train_clip_lora_coop.py --mode lora_only --data_dir ./data

# 仅CoOp模式
python train_clip_lora_coop.py --mode coop_only --data_dir ./data

# 使用配置文件
python train_clip_lora_coop.py -c config/clip_lora_coop/clip_lora_coop_full.yaml
```

### 2. 使用主训练脚本（与半监督算法结合）

```bash
# 与FixMatch结合
python train.py -c config/clip_lora_coop/clip_lora_coop_full.yaml
```

### 3. 零样本评估

```bash
# 使用专家提示
python eval_zeroshot_clip.py --prompt_mode expert --prompt_json ./semilearn/nets/clip_lora_coop/prompt.json

# 使用CLIP官方模板
python eval_zeroshot_clip.py --prompt_mode clip_full
```

## 配置参数说明

### CLIP模型设置
- `clip_model_name`: HuggingFace CLIP模型名称，默认 `openai/clip-vit-base-patch16`
- `prompt_json_path`: 类别描述JSON文件路径

### LoRA设置
- `use_lora`: 是否使用LoRA微调
- `lora_r`: LoRA秩，默认8
- `lora_alpha`: LoRA alpha参数，默认16
- `lora_dropout`: LoRA dropout率，默认0.1

### CoOp设置
- `use_coop`: 是否使用可学习软提示
- `num_context_tokens`: Context token数量，默认4
- `coop_temperature`: 温度系数，默认0.07

### 模式开关
- `freeze_vision`: 是否冻结视觉骨干网络
- `use_linear_probe`: 是否使用线性分类器模式

## 数据集格式

支持自定义文件夹结构：

```
data/
├── train/
│   ├── 01/  # 类别1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── 02/  # 类别2
│   │   ├── image1.jpg
│   ├── ...
├── val/
│   ├── 01/
│   ├── 02/
│   ├── ...
```

## prompt.json格式

```json
{
  "01": {
    "class_name": "Class Name 1",
    "description": "A detailed description of class 1..."
  },
  "02": {
    "class_name": "Class Name 2",
    "description": "A detailed description of class 2..."
  }
}
```

## 消融实验示例

```bash
# 1. 零样本基线
python train_clip_lora_coop.py --mode zero_shot

# 2. 仅LoRA
python train_clip_lora_coop.py --mode lora_only

# 3. 仅CoOp
python train_clip_lora_coop.py --mode coop_only

# 4. LoRA + CoOp (完整)
python train_clip_lora_coop.py --mode full

# 5. 大型LoRA (r=16)
python train_clip_lora_coop.py --mode large

# 6. 线性探测
python train_clip_lora_coop.py --mode linear_probe

# 7. 全参数微调
python train_clip_lora_coop.py --mode full_finetune
```

## 可训练参数统计

| 模式 | 参数量 | 占比 |
|------|--------|------|
| Full (ViT-B/16) | ~1.5M | ~2% |
| LoRA-only (r=8) | ~0.5M | ~0.7% |
| CoOp-only (n=4) | ~2K | ~0.003% |
| Linear Probe | ~768*9 | ~0.001% |

## 参考文献

- [CLIP](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [CoOp](https://arxiv.org/abs/2109.01134) - Learning to Prompt for Vision-Language Models