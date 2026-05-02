# Zero-shot CLIP Evaluation using project's built-in CLIP
# Supports both custom prompt.json and official CLIP templates
# Supports YAML config file

import os
import json
import argparse
import yaml
import numpy as np
import torch
from PIL import Image

# Use project's internal CLIP implementation
from semilearn.nets import clip


# CLIP official prompt templates (from OpenAI paper)
CLIP_OFFICIAL_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of my {}.",
    "a photo of {}.",
    "a bad photo of a {}.",
    "a bad photo of the {}.",
    "a bad photo of {}.",
    "a low resolution photo of a {}.",
    "a low resolution photo of the {}.",
    "a low resolution photo of {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a cropped photo of {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a bright photo of {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a dark photo of {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a blurry photo of {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a close-up photo of {}.",
    "a black and white photo of a {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {}.",
    "a pixelated photo of a {}.",
    "a pixelated photo of the {}.",
    "a pixelated photo of {}.",
    "a jpeg compressed photo of a {}.",
    "a jpeg compressed photo of the {}.",
    "a jpeg compressed photo of {}.",
    "a sketch of a {}.",
    "a sketch of the {}.",
    "a sketch of {}.",
    "a painting of a {}.",
    "a painting of the {}.",
    "a painting of {}.",
    "a drawing of a {}.",
    "a drawing of the {}.",
    "a drawing of {}.",
    "a rendition of a {}.",
    "a rendition of the {}.",
    "a rendition of {}.",
    "a sculpture of a {}.",
    "a sculpture of the {}.",
    "a sculpture of {}.",
    "an engraving of a {}.",
    "an engraving of the {}.",
    "an engraving of {}.",
    "a rendering of a {}.",
    "a rendering of the {}.",
    "a rendering of {}.",
]

# Simple prompt templates (commonly used)
SIMPLE_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def load_prompt_json(json_path):
    """Load class prompts from JSON file (detailed expert prompts)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sort by numeric key
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))

    prompts = []
    class_names = []

    for key in sorted_keys:
        info = data[key]
        class_names.append(info.get('class_name', key))
        # Build prompt: "{prefix} [Contour]: {Contour} [Pattern]: {Pattern}"
        prompt = f"{info['prefix']} [Contour]: {info['Contour']} [Pattern]: {info['Pattern']}"
        prompts.append(prompt)

    return class_names, prompts, sorted_keys


def build_clip_official_prompts(class_names, template_type='full'):
    """
    Build prompts using CLIP official templates.

    Args:
        class_names: List of class names
        template_type: 'full' (80 templates), 'simple' (3 templates), 'single' (1 template)

    Returns:
        prompts: List of prompts (multiple templates per class averaged)
        prompt_texts: List of all template texts for each class
    """
    if template_type == 'full':
        templates = CLIP_OFFICIAL_TEMPLATES
    elif template_type == 'simple':
        templates = SIMPLE_TEMPLATES
    else:  # single
        templates = ["a photo of a {}."]

    return templates


def encode_text_with_templates(model, class_names, templates, device):
    """
    Encode text features using multiple templates per class.
    Average the features across templates for more robust representations.

    Args:
        model: CLIP model
        class_names: List of class names
        templates: List of template strings with {} placeholder
        device: Device to use

    Returns:
        text_features: averaged text features [num_classes, embed_dim]
    """
    all_features = []

    with torch.no_grad():
        for template in templates:
            # Build prompts for this template
            prompts = [template.format(name) for name in class_names]
            text_tokens = clip.tokenize(prompts, truncate=True).to(device)

            # Encode
            features = model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features)

        # Stack and average
        all_features = torch.stack(all_features, dim=0)  # [num_templates, num_classes, embed_dim]
        text_features = all_features.mean(dim=0)  # [num_classes, embed_dim]
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def main():
    parser = argparse.ArgumentParser(description='Zero-shot CLIP Evaluation')

    # Config file (same as original USB system: -c/--c)
    parser.add_argument('-c', '--c', type=str, default=None,
                        dest='config',
                        help='Path to YAML config file')

    # Model settings (can be overridden by config)
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'],
                        help='CLIP model name')

    # Prompt settings
    parser.add_argument('--prompt_json', type=str, default=None,
                        help='Path to prompt.json (expert prompts)')
    parser.add_argument('--prompt_mode', type=str, default='expert',
                        choices=['expert', 'clip_full', 'clip_simple', 'clip_single'],
                        help='Prompt mode')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated class names')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data/imagenet/val',
                        help='Validation/test data directory')
    parser.add_argument('--batch_size', type=int, default=32)

    # Output settings
    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Override args with config values
        if 'model_name' in config:
            args.model_name = config['model_name']
        if 'prompt_mode' in config:
            args.prompt_mode = config['prompt_mode']
        if 'prompt_json' in config:
            args.prompt_json = config['prompt_json']
        if 'data_dir' in config:
            args.data_dir = config['data_dir']
        if 'batch_size' in config:
            args.batch_size = config['batch_size']
        if 'results_dir' in config:
            args.results_dir = config['results_dir']

        print(f"Loaded config from: {args.config}")

    print("=" * 60)
    print("Zero-shot CLIP Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Data: {args.data_dir}")
    print("=" * 60)

    # ========== 1. Load Model ==========
    print("\n[1] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_name, device=device)
    print(f"Model loaded on {device}")

    # ========== 2. Build Text Features ==========
    print("\n[2] Building text features...")

    class_names = None
    prompts = None
    class_ids = None

    if args.prompt_mode == 'expert':
        # Use detailed expert prompts from JSON
        if args.prompt_json is None:
            raise ValueError("--prompt_json is required for 'expert' mode")

        class_names, prompts, class_ids = load_prompt_json(args.prompt_json)
        print(f"Using expert prompts from: {args.prompt_json}")
        print(f"Classes ({len(class_names)}):")
        for cid, name in zip(class_ids, class_names):
            print(f"  {cid} -> {name}")
        print(f"\nExample prompt: {prompts[0][:100]}...")

        # Tokenize and encode
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

    else:
        # Use CLIP official templates
        # First get class names
        if args.prompt_json is not None:
            # Extract class names from JSON
            with open(args.prompt_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sorted_keys = sorted(data.keys(), key=lambda x: int(x))
            class_names = [data[key].get('class_name', key) for key in sorted_keys]
            class_ids = sorted_keys
        elif args.class_names is not None:
            class_names = args.class_names.split(',')
            class_ids = [str(i+1).zfill(2) for i in range(len(class_names))]
        else:
            # Default class names from folder names
            folders = sorted([f for f in os.listdir(args.data_dir)
                              if os.path.isdir(os.path.join(args.data_dir, f))])
            class_names = [f"Class {f}" for f in folders]
            class_ids = folders

        # Select template type
        if args.prompt_mode == 'clip_full':
            templates = build_clip_official_prompts(class_names, 'full')
            print(f"Using CLIP official templates: {len(templates)} templates per class")
        elif args.prompt_mode == 'clip_simple':
            templates = build_clip_official_prompts(class_names, 'simple')
            print(f"Using CLIP simple templates: {len(templates)} templates per class")
        else:
            templates = build_clip_official_prompts(class_names, 'single')
            print(f"Using single template: {templates[0]}")

        print(f"Classes ({len(class_names)}):")
        for cid, name in zip(class_ids, class_names):
            print(f"  {cid} -> {name}")

        # Encode with multiple templates
        text_features = encode_text_with_templates(model, class_names, templates, device)
        print(f"Text features shape: {text_features.shape}")

    # ========== 3. Load Dataset ==========
    print("\n[3] Loading dataset...")

    # Scan folders
    folders = sorted([f for f in os.listdir(args.data_dir)
                      if os.path.isdir(os.path.join(args.data_dir, f))])

    print(f"Found folders: {folders}")

    samples = []
    labels = []

    for folder in folders:
        folder_path = os.path.join(args.data_dir, folder)
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            samples.append(img_path)
            labels.append(int(folder) - 1)  # 01 -> 0, 02 -> 1, ...

    print(f"Total images: {len(samples)}")

    # ========== 4. Evaluate ==========
    print("\n[4] Running evaluation...")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Per-class stats
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for i in range(0, len(samples), args.batch_size):
            batch_paths = samples[i:i+args.batch_size]
            batch_labels = labels[i:i+args.batch_size]

            # Load and preprocess images
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                batch_images.append(preprocess(img))

            images_tensor = torch.stack(batch_images).to(device)

            # Encode images
            image_features = model.encode_image(images_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = similarity.argmax(dim=-1)

            # Statistics
            for j in range(len(preds)):
                pred = preds[j].item()
                label = batch_labels[j]

                total += 1
                if pred == label:
                    correct += 1

                all_preds.append(pred)
                all_labels.append(label)

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            if (i + args.batch_size) % 100 == 0 or i + args.batch_size >= len(samples):
                acc = correct / total if total > 0 else 0
                print(f"  Processed {min(i+args.batch_size, len(samples))}/{len(samples)}, Acc: {acc:.4f}")

    # ========== 5. Results ==========
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 60)
    print("Per-class Accuracy:")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {class_ids[i]} ({name}): {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    print("=" * 60)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)

    result_name = f'zeroshot_{args.prompt_mode}_results.npz'
    np.savez(
        os.path.join(args.results_dir, result_name),
        accuracy=accuracy,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        class_names=np.array(class_names),
        prompt_mode=args.prompt_mode
    )
    print(f"\nResults saved to {args.results_dir}/{result_name}")


if __name__ == "__main__":
    main()