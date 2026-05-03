# CLIP Linear Probe Training Script (Standalone)
# Supports: Linear Probe, Zero-shot evaluation, Fine-tuning modes
# No dependency on semilearn package chain imports

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Direct imports from clip module (avoid semilearn chain imports)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CLIP directly without triggering semilearn.__init__
_clip_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'semilearn', 'nets', 'clip')
sys.path.insert(0, _clip_module_path)

from clip import load as clip_load, tokenize as clip_tokenize


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make CUDA operations deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Random seed set to {seed} for reproducibility")


class SimpleImageDataset(torch.utils.data.Dataset):
    """Simple dataset for numeric folder structure."""

    def __init__(self, data_dir, split='train', num_labels_per_class=None, transform=None, seed=0):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Determine which split to use
        if split == 'train':
            folder_path = os.path.join(data_dir, 'train')
        elif split == 'val' or split == 'test':
            folder_path = os.path.join(data_dir, 'val')
        else:
            folder_path = data_dir

        # Check if folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Data folder not found: {folder_path}")

        # Scan folders
        folders = sorted([f for f in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, f))])

        for folder in folders:
            class_path = os.path.join(folder_path, folder)
            images = sorted([f for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            # For training with limited labels, select subset
            if split == 'train' and num_labels_per_class is not None:
                np.random.shuffle(images)
                images = images[:num_labels_per_class]

            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                self.samples.append(img_path)
                self.labels.append(int(folder) - 1)  # 01 -> 0, 02 -> 1, ...

        print(f"[{split}] Loaded {len(self.samples)} images from {len(folders)} classes")
        if split == 'train' and num_labels_per_class is not None:
            print(f"[{split}] Using {num_labels_per_class} samples per class")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    parser = argparse.ArgumentParser(description='CLIP Linear Probe Training')

    # Config file
    parser.add_argument('-c', '--c', type=str, default=None,
                        dest='config',
                        help='Path to YAML config file')

    # Model settings
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'],
                        help='CLIP backbone model')
    parser.add_argument('--mode', type=str, default='linear_probe',
                        choices=['linear_probe', 'zero_shot'],
                        help='Training mode')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data/imagenet',
                        help='Data directory')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes')
    parser.add_argument('--num_labels_per_class', type=int, default=25,
                        help='Number of labeled samples per class (for few-shot)')
    parser.add_argument('--num_labels', type=int, default=None,
                        help='Total number of labeled samples (alternative to per_class)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)

    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'SGD', 'Adam'])

    # Output settings
    parser.add_argument('--save_dir', type=str, default='./saved_models/clip_linear_probe')
    parser.add_argument('--save_name', type=str, default='clip_linear_probe')
    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        print(f"Loaded config from: {args.config}")

    # ========== Set Random Seed for Reproducibility ==========
    set_random_seed(args.seed)

    # Calculate num_labels if using per_class
    if args.num_labels is None:
        args.num_labels = args.num_labels_per_class * args.num_classes

    print("=" * 60)
    print("CLIP Linear Probe Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Labeled samples: {args.num_labels} ({args.num_labels_per_class}/class)")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print("=" * 60)

    # ========== 1. Load CLIP Model ==========
    print("\n[1] Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip_load(args.model_name, device=device)

    # Freeze visual encoder
    for param in clip_model.visual.parameters():
        param.requires_grad = False

    print(f"CLIP loaded on {device}")
    print(f"Visual encoder frozen")

    # ========== 2. Create Linear Classifier ==========
    print("\n[2] Creating linear classifier...")

    # Get visual feature dimension
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        dummy_feature = clip_model.encode_image(dummy_image)
        feature_dim = dummy_feature.shape[-1]

    print(f"Feature dimension: {feature_dim}")

    classifier = nn.Linear(feature_dim, args.num_classes).to(device)
    print(f"Classifier: Linear({feature_dim}, {args.num_classes})")

    # ========== 3. Load Dataset ==========
    print("\n[3] Loading dataset...")

    # Training set (limited labels)
    train_dataset = SimpleImageDataset(
        args.data_dir,
        split='train',
        num_labels_per_class=args.num_labels_per_class,
        transform=preprocess,
        seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Validation set (all samples)
    val_dataset = SimpleImageDataset(
        args.data_dir,
        split='val',
        transform=preprocess
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # ========== 4. Setup Training ==========
    print("\n[4] Setting up training...")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print(f"Optimizer: {args.optimizer}")
    print(f"Loss: CrossEntropyLoss")

    # ========== 5. Training Loop ==========
    print("\n[5] Training...")
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        classifier.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Extract features (frozen)
            with torch.no_grad():
                features = clip_model.encode_image(images)

            # Linear classification
            logits = classifier(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                features = clip_model.encode_image(images)
                logits = classifier(features)
                _, preds = logits.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'{args.save_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'clip_model_name': args.model_name,
                'feature_dim': feature_dim,
                'num_classes': args.num_classes,
                'val_acc': val_acc,
            }, save_path)
            print(f"  -> Saved best model (Val Acc: {val_acc:.4f})")

    # ========== 6. Final Results ==========
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Best Epoch: {best_epoch}")
    print(f"Model saved: {args.save_dir}/{args.save_name}_best.pt")
    print("=" * 60)

    # ========== 7. Final Evaluation ==========
    print("\n[6] Loading best model for final evaluation...")

    # Load best checkpoint
    checkpoint = torch.load(os.path.join(args.save_dir, f'{args.save_name}_best.pt'))
    classifier.load_state_dict(checkpoint['classifier'])

    classifier.eval()
    all_preds = []
    all_labels = []

    # Per-class accuracy
    class_correct = [0] * args.num_classes
    class_total = [0] * args.num_classes

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            features = clip_model.encode_image(images)
            logits = classifier(features)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(preds)):
                label = labels[i].item()
                if preds[i] == labels[i]:
                    class_correct[label] += 1
                class_total[label] += 1

    print("\nPer-class Accuracy:")
    for i in range(args.num_classes):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  Class {i+1:02d}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    np.savez(
        os.path.join(args.results_dir, f'{args.save_name}_results.npz'),
        best_acc=best_acc,
        best_epoch=best_epoch,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        class_correct=np.array(class_correct),
        class_total=np.array(class_total),
        num_labels_per_class=args.num_labels_per_class
    )
    print(f"\nResults saved to {args.results_dir}/{args.save_name}_results.npz")


if __name__ == "__main__":
    main()