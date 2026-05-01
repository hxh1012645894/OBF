# Simple Zero-shot CLIP Evaluation using open_clip
# Adapted for numeric class folder structure (01, 02, ..., 09)

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
import open_clip
from torch.utils.data import DataLoader, Dataset


class SimpleImageDataset(Dataset):
    """Simple dataset for numeric folder structure."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_ids = []

        # Scan folders (01, 02, ..., 09)
        folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

        for folder in folders:
            folder_path = os.path.join(data_dir, folder)
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                self.samples.append(img_path)
                # Convert folder name to class index (01 -> 0, 02 -> 1, etc.)
                self.class_ids.append(int(folder) - 1)  # 01 -> 0, 02 -> 1, ..., 09 -> 8

        print(f"Loaded {len(self.samples)} images from {len(folders)} classes")
        print(f"Classes: {folders}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.class_ids[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def load_prompt_json(json_path):
    """Load class prompts from JSON file.

    JSON format (numeric keys matching folder names):
    {
        "01": {"class_name": "Left Epiplastron", "prefix": "...", "Contour": "...", "Pattern": "..."},
        "02": {...},
        ...
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sort by numeric key to ensure correct order
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


def main():
    parser = argparse.ArgumentParser(description='Zero-shot CLIP Evaluation')

    # Model settings
    parser.add_argument('--model_name', type=str, default='ViT-B-32',
                        choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-g-14'],
                        help='OpenCLIP model name')
    parser.add_argument('--pretrained', type=str, default='openai',
                        help='Pretrained weights (openai, laion2b_s34b_b79k, etc.)')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data/imagenet/val',
                        help='Validation/test data directory')
    parser.add_argument('--prompt_json', type=str, required=True,
                        help='Path to prompt.json with class descriptions')
    parser.add_argument('--batch_size', type=int, default=32)

    # Output settings
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--save_predictions', type=bool, default=True)

    args = parser.parse_args()

    print("=" * 60)
    print("Zero-shot CLIP Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name} ({args.pretrained})")
    print(f"Data: {args.data_dir}")
    print(f"Prompt: {args.prompt_json}")
    print("=" * 60)

    # ========== 1. Load Model ==========
    print("\n[1] Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # ========== 2. Build Text Features ==========
    print("\n[2] Building text features...")
    class_names, prompts, class_ids = load_prompt_json(args.prompt_json)

    print(f"Classes ({len(class_names)}):")
    for i, (cid, name) in enumerate(zip(class_ids, class_names)):
        print(f"  {cid} -> {name}")

    print(f"\nExample prompt: {prompts[0][:100]}...")

    # Tokenize and encode
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Text features shape: {text_features.shape}")

    # ========== 3. Load Dataset ==========
    print("\n[3] Loading dataset...")
    dataset = SimpleImageDataset(args.data_dir, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ========== 4. Evaluate ==========
    print("\n[4] Running evaluation...")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_paths = []

    # Per-class accuracy
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            # Encode image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with text features
            similarity = (100.0 * image_features @ text_features.T)
            probs = similarity.softmax(dim=-1)
            preds = probs.argmax(dim=-1)

            # Statistics
            for i in range(len(preds)):
                pred = preds[i].item()
                label = labels[i].item()

                total += 1
                if pred == label:
                    correct += 1

                all_preds.append(pred)
                all_labels.append(label)
                all_paths.append(paths[i])

                # Per-class stats
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(loader) - 1:
                acc = correct / total if total > 0 else 0
                print(f"  Batch {batch_idx + 1}/{len(loader)}, Running Acc: {acc:.4f}")

    # ========== 5. Results ==========
    accuracy = correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_name} ({args.pretrained})")
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

    # ========== 6. Save Results ==========
    os.makedirs(args.results_dir, exist_ok=True)

    results_file = os.path.join(args.results_dir, 'zeroshot_results.npz')
    np.savez(
        results_file,
        accuracy=accuracy,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        class_names=np.array(class_names),
        class_ids=np.array(class_ids),
        image_paths=np.array(all_paths)
    )
    print(f"\nResults saved to: {results_file}")

    # Also save a readable text report
    report_file = os.path.join(args.results_dir, 'zeroshot_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Zero-shot CLIP Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_name} ({args.pretrained})\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Prompt: {args.prompt_json}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write("-" * 60 + "\n")
        f.write("Per-class Accuracy:\n")
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                f.write(f"  {class_ids[i]} ({name}): {class_acc:.4f} ({class_correct[i]}/{class_total[i]})\n")
        f.write("=" * 60 + "\n")
    print(f"Report saved to: {report_file}")

    return accuracy


if __name__ == "__main__":
    main()