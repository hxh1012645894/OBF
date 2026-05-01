# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Eval script for CLIPLoRACoOp models (Zero-shot, LoRA, CoOp, etc.)

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from semilearn.core.utils import get_dataset
from semilearn.nets import clip_zero_shot, clip_lora_coop, clip_coop_only, clip_lora_only


def get_clip_model(model_type: str, num_classes: int, prompt_json_path: str, **kwargs):
    """Get CLIP model by type name."""
    model_factories = {
        'clip_zero_shot': clip_zero_shot,
        'clip_lora_coop': clip_lora_coop,
        'clip_coop_only': clip_coop_only,
        'clip_lora_only': clip_lora_only,
    }

    factory = model_factories.get(model_type, clip_lora_coop)
    return factory(
        num_classes=num_classes,
        prompt_json_path=prompt_json_path,
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIPLoRACoOp models')

    # Model configurations
    parser.add_argument('--model_type', type=str, default='clip_zero_shot',
                        choices=['clip_zero_shot', 'clip_lora_coop', 'clip_coop_only', 'clip_lora_only'],
                        help='Type of CLIP model to evaluate')
    parser.add_argument('--prompt_json_path', type=str, required=True,
                        help='Path to prompt.json file with class descriptions')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (optional, for trained models)')

    # LoRA/CoOp configurations (for non-zero-shot modes)
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--use_coop', type=bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--num_context_tokens', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.07)

    # Data configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    # Evaluation settings
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("CLIP Model Evaluation Configuration")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Prompt JSON: {args.prompt_json_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Num classes: {args.num_classes}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Build model
    print("\n[1] Building model...")
    model = get_clip_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        prompt_json_path=args.prompt_json_path,
        use_lora=args.use_lora if args.model_type != 'clip_zero_shot' else False,
        use_coop=args.use_coop if args.model_type != 'clip_zero_shot' else False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_context_tokens=args.num_context_tokens,
        temperature=args.temperature
    )

    # Load checkpoint if provided
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"\n[2] Loading checkpoint from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'ema_model' in checkpoint:
            state_dict = checkpoint['ema_model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
                new_state_dict[new_key] = item
            else:
                new_state_dict[key] = item

        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print("\n[2] No checkpoint provided, using pretrained CLIP weights.")

    # Move to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device}")

    # Build dataset
    print("\n[3] Building dataset...")
    # Create dummy args for get_dataset
    class DummyArgs:
        def __init__(self, real_args):
            self.data_dir = real_args.data_dir
            self.dataset = real_args.dataset
            self.num_classes = real_args.num_classes
            self.num_labels = 40  # dummy value
            self.ulb_num_labels = 5000  # dummy value
            self.lb_imb_ratio = 1
            self.ulb_imb_ratio = 1
            self.seed = 0
            self.epoch = 1
            self.num_train_iter = 1024
            self.img_size = real_args.img_size
            self.crop_ratio = real_args.crop_ratio
            self.max_length = 512
            self.max_length_seconds = 4.0
            self.sample_rate = 16000
            self.include_lb_to_ulb = False

    dummy_args = DummyArgs(args)
    dataset_dict = get_dataset(dummy_args, 'fixmatch', args.dataset, args.num_labels,
                               args.num_classes, args.data_dir, False)

    if 'eval' in dataset_dict:
        eval_dataset = dataset_dict['eval']
    elif 'test' in dataset_dict and dataset_dict['test'] is not None:
        eval_dataset = dataset_dict['test']
    else:
        raise ValueError(f"No evaluation dataset found in dataset_dict")

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # Evaluation loop
    print("\n[4] Running evaluation...")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_loader):
            images = data['x_lb']
            labels = data['y_lb']

            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            # Statistics
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(eval_loader)}, "
                      f"Running Acc: {correct/total:.4f}")

    # Final results
    accuracy = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    # Save results
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(
            args.results_dir,
            f"{args.model_type}_{args.dataset}_results.npz"
        )
        np.savez(
            results_file,
            accuracy=accuracy,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs
        )
        print(f"\nResults saved to: {results_file}")

    return accuracy


if __name__ == "__main__":
    main()