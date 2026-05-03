# CoOp (Context Optimization) Training Script
# Learnable soft prompts with frozen CLIP backbone
# Supports expert prompts from prompt.json

import os
import json
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
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Random seed set to {seed} for reproducibility")


class CoOpModel(nn.Module):
    """
    CoOp: Context Optimization for CLIP.

    Key idea: Learn soft prompt tokens (context) while keeping CLIP frozen.
    Prompt format: [SOS] [ctx_1] [ctx_2] ... [ctx_M] [class_tokens] [EOS]

    Supports two modes:
    1. 'expert': Use detailed prompts from prompt.json (Contour + Pattern)
    2. 'simple': Use simple "a photo of {class_name}" template
    """

    def __init__(
        self,
        clip_model,
        num_classes: int,
        num_context_tokens: int = 4,
        class_names: list = None,
        prompt_json_path: str = None,
        device: str = 'cuda'
    ):
        super().__init__()

        self.clip_model = clip_model
        self.num_classes = num_classes
        self.num_context_tokens = num_context_tokens
        self.device = device

        # Freeze ALL CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get embedding dimension from visual encoder
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            visual_feat = self.clip_model.encode_image(dummy)
        self.embed_dim = visual_feat.shape[-1]

        # ========== Learnable Context Tokens (CoOp) ==========
        # Initialize with random normal (following CoOp paper)
        # Shape: [num_context_tokens, embed_dim]
        self.context_tokens = nn.Parameter(
            torch.randn(num_context_tokens, self.embed_dim) * 0.02
        )

        # Alternative: Initialize from word embeddings (optional)
        # self._init_context_from_words("a photo of a")  # Uncomment to use

        # ========== Build Class Prompts ==========
        self.class_names = class_names
        self.prompt_data = None

        if prompt_json_path and os.path.exists(prompt_json_path):
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                self.prompt_data = json.load(f)
            # Sort by numeric key
            sorted_keys = sorted(self.prompt_data.keys(), key=lambda x: int(x))
            self.class_names = [self.prompt_data[k].get('class_name', k) for k in sorted_keys]
            print(f"[CoOp] Loaded expert prompts from: {prompt_json_path}")

        if self.class_names is None:
            raise ValueError("Either class_names or prompt_json_path must be provided")

        # Pre-compute class token embeddings (frozen part after context)
        self._prepare_class_token_embeddings()

        print(f"[CoOp] Initialized with {num_context_tokens} context tokens")
        print(f"[CoOp] Embedding dimension: {self.embed_dim}")
        print(f"[CoOp] Trainable parameters: {num_context_tokens * self.embed_dim}")

    def _init_context_from_words(self, init_words: str = "a photo of a"):
        """
        Initialize context tokens from real word embeddings.
        This can help accelerate convergence compared to random initialization.

        Args:
            init_words: Words to use for initialization (e.g., "a photo of a")
        """
        tokens = clip_tokenize([init_words], truncate=True).to(self.device)

        with torch.no_grad():
            word_embeds = self.clip_model.token_embedding(tokens[0])

            # Skip SOS (position 0), take next N tokens
            # word_embeds[1] = "a", word_embeds[2] = "photo", etc.
            num_words = min(self.num_context_tokens, len(init_words.split()) + 1)

            # Initialize context tokens from word embeddings
            init_embeds = word_embeds[1:num_words+1, :].clone()

            # If we need more tokens than words, pad with random
            if num_words < self.num_context_tokens:
                padding = torch.randn(self.num_context_tokens - num_words, self.embed_dim) * 0.02
                init_embeds = torch.cat([init_embeds, padding], dim=0)

            # Assign to context_tokens
            self.context_tokens.data = init_embeds.to(self.device)

        print(f"[CoOp] Initialized context tokens from words: '{init_words}'")

    def _prepare_class_token_embeddings(self):
        """
        Pre-compute token embeddings for class descriptions (frozen part).
        These will be concatenated with learnable context tokens.
        """
        self.class_token_embeds_list = []

        for i, class_name in enumerate(self.class_names):
            # Build prompt text
            if self.prompt_data:
                # Use expert prompts
                key = str(i + 1).zfill(2)  # "01", "02", ...
                if key in self.prompt_data:
                    info = self.prompt_data[key]
                    # Format: "{prefix} [Contour]: {Contour} [Pattern]: {Pattern}"
                    prompt_text = f"{info['prefix']} [Contour]: {info['Contour']} [Pattern]: {info['Pattern']}"
                else:
                    prompt_text = f"a photo of {class_name}"
            else:
                # Simple template
                prompt_text = f"a photo of {class_name}"

            # Tokenize
            tokens = clip_tokenize([prompt_text], truncate=True).to(self.device)

            # Get token embeddings (excluding SOS)
            with torch.no_grad():
                # CLIP's tokenize output: [SOS, word_tokens, EOS, PAD...]
                # We need: [word_tokens, EOS] to append after context
                # Note: token_embeds[0] is SOS, we'll handle it separately

                # Get full token embeddings
                token_embeds = self.clip_model.token_embedding(tokens)

                # Store: we'll use tokens from position 1 onwards (after SOS)
                # Find EOS position (token id 49407 for CLIP)
                eos_pos = (tokens[0] == 49407).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    eos_idx = eos_pos[0].item()
                else:
                    eos_idx = 76  # Default to max length - 1

                # Store embeddings from position 1 to EOS (inclusive)
                # Shape: [seq_len, embed_dim]
                class_embeds = token_embeds[0, 1:eos_idx+1, :].clone()  # [word_tokens + EOS]
                self.class_token_embeds_list.append(class_embeds)

        print(f"[CoOp] Prepared token embeddings for {len(self.class_names)} classes")

    def forward(self, images):
        """
        Forward pass: encode images, compute text features with context, classify.

        Args:
            images: [B, 3, 224, 224]

        Returns:
            logits: [B, num_classes]
        """
        # ========== Encode Images (frozen) ==========
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        # ========== Encode Text with Context (learnable) ==========
        text_features = self._encode_text_with_context()

        # ========== Compute Similarity ==========
        # image_features: [B, embed_dim]
        # text_features: [num_classes, embed_dim]
        logits = (image_features @ text_features.T) * 100.0  # CLIP's temperature scale

        return logits

    def _encode_text_with_context(self):
        """
        Encode text with learnable context tokens prepended.

        Format: [SOS] [ctx_1] ... [ctx_M] [class_tokens] [EOS]

        Returns:
            text_features: [num_classes, embed_dim]
        """
        text_features_list = []

        # Get SOS embedding
        sos_token = clip_tokenize([""], truncate=True).to(self.device)
        with torch.no_grad():
            sos_embed = self.clip_model.token_embedding(sos_token)[0, 0:1, :]  # [1, embed_dim]

        for class_embeds in self.class_token_embeds_list:
            # Concatenate: SOS + context + class_tokens
            # context_tokens: [num_ctx, embed_dim]
            # class_embeds: [class_seq_len, embed_dim]

            # Truncate class tokens if too long (77 = SOS + ctx + class + EOS)
            max_class_len = 77 - 1 - self.num_context_tokens - 1
            class_embeds_truncated = class_embeds[:max_class_len]

            # Build combined sequence
            combined = torch.cat([
                sos_embed,                     # [1, embed_dim]
                self.context_tokens,           # [num_ctx, embed_dim]
                class_embeds_truncated         # [class_len, embed_dim]
            ], dim=0)  # [seq_len, embed_dim]

            # Add positional embeddings
            seq_len = combined.shape[0]
            with torch.no_grad():
                pos_embed = self.clip_model.positional_embedding[:seq_len]
                combined = combined + pos_embed

            # Pass through transformer
            combined = combined.unsqueeze(0)  # [1, seq_len, embed_dim]

            with torch.no_grad():
                # CLIP text encoder layers
                x = combined.type(self.clip_model.conv1.weight.dtype)
                x = self.clip_model.ln_final(x)

                # Take EOS token representation (last token)
                text_feat = x[0, -1, :] @ self.clip_model.text_projection

            text_feat = F.normalize(text_feat, dim=-1)
            text_features_list.append(text_feat)

        # Stack all class features
        text_features = torch.stack(text_features_list, dim=0)  # [num_classes, embed_dim]

        return text_features

    def get_trainable_params(self):
        """Return only the context tokens as trainable parameters."""
        return [self.context_tokens]


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
        elif split == 'val' or split == 'test':
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


def main():
    parser = argparse.ArgumentParser(description='CoOp Training')

    # Config file
    parser.add_argument('-c', '--c', type=str, default=None, dest='config')

    # Model settings
    parser.add_argument('--model_name', type=str, default='ViT-B/32')
    parser.add_argument('--num_context_tokens', type=int, default=4, help='Number of learnable context tokens')
    parser.add_argument('--init_method', type=str, default='random', choices=['random', 'words'],
                        help='Context token initialization: random or words (from "a photo of a")')
    parser.add_argument('--init_words', type=str, default='a photo of a',
                        help='Words to use for initialization when init_method=words')

    # Prompt settings
    parser.add_argument('--prompt_json', type=str, default=None, help='Path to prompt.json (expert prompts)')
    parser.add_argument('--class_names', type=str, default=None, help='Comma-separated class names')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--num_labels_per_class', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)

    # Training settings
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW', 'Adam'])

    # Output settings
    parser.add_argument('--save_dir', type=str, default='./saved_models/coop')
    parser.add_argument('--save_name', type=str, default='coop_obf')
    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        print(f"Loaded config from: {args.config}")

    # ========== Set Random Seed for Reproducibility ==========
    set_random_seed(args.seed)

    print("=" * 60)
    print("CoOp (Context Optimization) Training")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Context tokens: {args.num_context_tokens}")
    print(f"Data: {args.data_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Labeled samples: {args.num_labels_per_class}/class")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print("=" * 60)

    # ========== 1. Load CLIP ==========
    print("\n[1] Loading CLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = clip_load(args.model_name, device=device)
    print(f"CLIP loaded on {device}")

    # ========== 2. Create CoOp Model ==========
    print("\n[2] Creating CoOp model...")

    # Get class names from prompt_json if available
    class_names = None
    if args.class_names:
        class_names = args.class_names.split(',')

    coop_model = CoOpModel(
        clip_model=clip_model,
        num_classes=args.num_classes,
        num_context_tokens=args.num_context_tokens,
        class_names=class_names,
        prompt_json_path=args.prompt_json,
        device=device
    ).to(device)

    # Initialize context tokens from words if specified
    if args.init_method == 'words':
        coop_model._init_context_from_words(args.init_words)

    # Print trainable parameters
    trainable = sum(p.numel() for p in coop_model.get_trainable_params())
    total = sum(p.numel() for p in coop_model.parameters())
    print(f"Trainable: {trainable} / {total} = {100*trainable/total:.2f}%")

    # ========== 3. Load Dataset ==========
    print("\n[3] Loading dataset...")

    train_dataset = SimpleImageDataset(
        args.data_dir, split='train',
        num_labels_per_class=args.num_labels_per_class,
        transform=preprocess, seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = SimpleImageDataset(
        args.data_dir, split='val',
        transform=preprocess
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # ========== 4. Setup Training ==========
    print("\n[4] Setting up training...")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(coop_model.get_trainable_params(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(coop_model.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(coop_model.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Optimizer: {args.optimizer}")

    # ========== 5. Training Loop ==========
    print("\n[5] Training...")
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        coop_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = coop_model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})

        train_acc = train_correct / train_total

        # Validation
        coop_model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = coop_model(images)
                _, preds = logits.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'{args.save_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'context_tokens': coop_model.context_tokens.data,
                'num_context_tokens': args.num_context_tokens,
                'model_name': args.model_name,
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

    # ========== 7. Per-class Accuracy ==========
    print("\n[6] Final evaluation...")

    coop_model.eval()
    class_correct = [0] * args.num_classes
    class_total = [0] * args.num_classes
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = coop_model(images)
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
            acc = class_correct[i] / class_total[i]
            print(f"  Class {i+1:02d}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    np.savez(
        os.path.join(args.results_dir, f'{args.save_name}_results.npz'),
        best_acc=best_acc,
        best_epoch=best_epoch,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        context_tokens=coop_model.context_tokens.data.cpu().numpy(),
    )
    print(f"\nResults saved to {args.results_dir}/{args.save_name}_results.npz")


if __name__ == "__main__":
    main()