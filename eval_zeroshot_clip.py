# Zero-shot CLIP Evaluation using project's built-in CLIP
# No external dependencies needed!

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image

# Use project's internal CLIP implementation
from semilearn.nets import clip


def load_prompt_json(json_path):
    """Load class prompts from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sort by numeric key
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))

    prompts = []
    class_names = []

    for key in sorted_keys:
        info = data[key]
        class_names.append(info.get('class_name', key))
        # Build prompt
        prompt = f"{info['prefix']} [Contour]: {info['Contour']} [Pattern]: {info['Pattern']}"
        prompts.append(prompt)

    return class_names, prompts, sorted_keys


def main():
    parser = argparse.ArgumentParser(description='Zero-shot CLIP Evaluation')

    # Model settings
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'],
                        help='CLIP model name')

    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data/imagenet/val',
                        help='Validation/test data directory')
    parser.add_argument('--prompt_json', type=str, required=True,
                        help='Path to prompt.json')
    parser.add_argument('--batch_size', type=int, default=32)

    # Output settings
    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    print("=" * 60)
    print("Zero-shot CLIP Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Prompt: {args.prompt_json}")
    print("=" * 60)

    # ========== 1. Load Model ==========
    print("\n[1] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_name, device=device)
    print(f"Model loaded on {device}")
    print(f"Available models: {clip.available_models()}")

    # ========== 2. Build Text Features ==========
    print("\n[2] Building text features...")
    class_names, prompts, class_ids = load_prompt_json(args.prompt_json)

    print(f"Classes ({len(class_names)}):")
    for cid, name in zip(class_ids, class_names):
        print(f"  {cid} -> {name}")

    print(f"\nExample prompt: {prompts[0][:80]}...")

    # Tokenize prompts
    text_tokens = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

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
            labels_tensor = torch.tensor(batch_labels).to(device)

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
    np.savez(
        os.path.join(args.results_dir, 'zeroshot_results.npz'),
        accuracy=accuracy,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        class_names=np.array(class_names)
    )
    print(f"\nResults saved to {args.results_dir}/zeroshot_results.npz")


if __name__ == "__main__":
    main()