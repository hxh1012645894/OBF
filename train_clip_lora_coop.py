#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# CLIP + LoRA + CoOp Training Script
# Standalone training script for CLIP-based models with configurable modes

import os
import sys
import argparse
import yaml
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semilearn.nets.clip_lora_coop import (
    CLIPLoRACoOp,
    clip_lora_coop,
    clip_lora_coop_small,
    clip_lora_coop_large,
    clip_zero_shot,
    clip_linear_probe,
    clip_coop_only,
    clip_lora_only,
    clip_full_finetune
)


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


def setup_logging(save_dir: str, save_name: str):
    """Setup logging to file and console."""
    log_dir = os.path.join(save_dir, save_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'log.txt')
    logger = logging.getLogger('clip_lora_coop_train')
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class SimpleImageDataset(torch.utils.data.Dataset):
    """Simple dataset for numeric folder structure."""

    def __init__(self, data_dir, split='train', num_labels_per_class=None, transform=None, seed=0):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        np.random.seed(seed)

        if split == 'train':
            folder_path = os.path.join(data_dir, 'train')
        elif split in ['val', 'test']:
            folder_path = os.path.join(data_dir, 'val')
        else:
            folder_path = data_dir

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Data folder not found: {folder_path}")

        folders = sorted([f for f in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, f))])

        for folder in folders:
            class_path = os.path.join(folder_path, folder)
            images = sorted([f for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            if split == 'train' and num_labels_per_class is not None:
                np.random.shuffle(images)
                images = images[:num_labels_per_class]

            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                self.samples.append(img_path)
                self.labels.append(int(folder) - 1)

        print(f"[{split}] Loaded {len(self.samples)} images from {len(folders)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def get_clip_transforms(img_size=224):
    """Get CLIP-compatible transforms."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


def get_model_by_mode(mode, args):
    """Get model based on mode configuration."""
    model_map = {
        'full': clip_lora_coop,
        'lora_only': clip_lora_only,
        'coop_only': clip_coop_only,
        'zero_shot': clip_zero_shot,
        'linear_probe': clip_linear_probe,
        'large': clip_lora_coop_large,
        'small': clip_lora_coop_small,
        'full_finetune': clip_full_finetune,
    }

    if mode in model_map:
        return model_map[mode](
            num_classes=args.num_classes,
            pretrained=True,
            prompt_json_path=args.prompt_json_path,
            clip_model_name=args.clip_model_name,
            use_lora=args.use_lora,
            use_coop=args.use_coop,
            freeze_vision=args.freeze_vision,
            use_linear_probe=args.use_linear_probe,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_context_tokens=args.num_context_tokens,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(model_map.keys())}")


def main():
    parser = argparse.ArgumentParser(description='CLIP + LoRA + CoOp Training')

    # Config file
    parser.add_argument('-c', '--c', type=str, default=None, dest='config')

    # Mode selection (preset configurations)
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'lora_only', 'coop_only', 'zero_shot',
                                 'linear_probe', 'large', 'small', 'full_finetune'],
                        help='Training mode preset')

    # CLIP model settings
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--prompt_json_path', type=str, default='./semilearn/nets/clip_lora_coop/prompt.json')

    # LoRA settings
    parser.add_argument('--use_lora', type=str2bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # CoOp settings
    parser.add_argument('--use_coop', type=str2bool, default=True)
    parser.add_argument('--num_context_tokens', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.07)

    # Mode switches
    parser.add_argument('--freeze_vision', type=str2bool, default=True)
    parser.add_argument('--use_linear_probe', type=str2bool, default=False)

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--num_labels_per_class', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)

    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW', 'Adam'])

    # Output settings
    parser.add_argument('--save_dir', type=str, default='./saved_models/clip_lora_coop')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='./results')

    # Logging
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        print(f"Loaded config from: {args.config}")

    # Set default save name based on mode
    if args.save_name is None:
        args.save_name = f"clip_{args.mode}"

    # Set random seed
    set_random_seed(args.seed)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CLIP + LoRA + CoOp Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"CLIP model: {args.clip_model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Labeled samples: {args.num_labels_per_class}/class")
    print(f"Device: {device}")
    print("=" * 60)

    # ========== 1. Create Model ==========
    print("\n[1] Creating model...")
    model = get_model_by_mode(args.mode, args).to(device)
    model.print_trainable_parameters()

    # ========== 2. Create Dataset ==========
    print("\n[2] Loading dataset...")
    transform = get_clip_transforms(args.img_size)

    train_dataset = SimpleImageDataset(
        args.data_dir, split='train',
        num_labels_per_class=args.num_labels_per_class,
        transform=transform, seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = SimpleImageDataset(
        args.data_dir, split='val',
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # ========== 3. Setup Training ==========
    print("\n[3] Setting up training...")

    criterion = nn.CrossEntropyLoss()

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")

    # Setup logging
    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
    logger = setup_logging(args.save_dir, args.save_name)

    if args.use_tensorboard:
        tb_dir = os.path.join(args.save_dir, args.save_name, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        print(f"[TensorBoard] Logging to {tb_dir}")
    else:
        writer = None

    # ========== 4. Training Loop ==========
    print("\n[4] Training...")
    logger.info("=" * 60)
    logger.info(f"Training started - Mode: {args.mode}")
    logger.info("=" * 60)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            logits = outputs['logits']
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                logits = outputs['logits']
                _, preds = logits.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total

        # Logging
        log_msg = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
        print(log_msg)
        logger.info(log_msg)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

            save_path = os.path.join(args.save_dir, args.save_name, 'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'args': vars(args),
            }, save_path)
            logger.info(f"  -> Saved best model (Val Acc: {val_acc:.4f})")

    # ========== 5. Final Results ==========
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Best Epoch: {best_epoch}")
    print("=" * 60)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Val Accuracy: {best_acc:.4f}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info("=" * 60)

    if writer is not None:
        writer.close()


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()