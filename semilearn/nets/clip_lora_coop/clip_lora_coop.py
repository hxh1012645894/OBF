# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# OBF-MANet-V2: Multi-modal Semi-supervised Learning with CLIP + LoRA + CoOp

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class CLIPLoRACoOp(nn.Module):
    """
    CLIP with LoRA fine-tuning on visual encoder and CoOp (Context Optimization) on text encoder.

    This model combines:
    1. LoRA adapters on visual encoder's attention layers (q_proj, v_proj)
    2. Learnable soft prompts (CoOp) prepended to text prompts
    3. Hard prompts constructed from class-specific attributes (contour, pattern)

    Args:
        num_classes (int): Number of classes for classification
        lora_r (int): LoRA rank (default: 8)
        lora_alpha (int): LoRA alpha (default: 16)
        lora_dropout (float): LoRA dropout rate (default: 0.1)
        num_context_tokens (int): Number of learnable context tokens for CoOp (default: 4)
        temperature (float): Temperature for cosine similarity scaling (default: 0.07)
        pretrained_model_name (str): HuggingFace CLIP model name (default: 'openai/clip-vit-base-patch16')
        prompt_json_path (str): Path to prompt.json file containing class attributes
        device (str): Device to load model on (default: 'cuda' if available)
    """

    def __init__(
        self,
        num_classes: int,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_context_tokens: int = 4,
        temperature: float = 0.07,
        pretrained_model_name: str = 'openai/clip-vit-base-patch16',
        prompt_json_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_context_tokens = num_context_tokens
        self.temperature = temperature
        self.pretrained_model_name = pretrained_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ========== 1. Load Pretrained CLIP Model ==========
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name)

        # Get embedding dimension from CLIP config
        self.embed_dim = self.clip_model.config.text_config.hidden_size  # Typically 512 for clip-vit-base

        # ========== 2. Visual Branch - LoRA Fine-tuning ==========
        # Freeze all visual encoder parameters
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False

        # Configure LoRA for visual encoder attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_proj', 'v_proj'],  # Target Query and Value projections
            modules_to_save=None,
        )

        # Apply LoRA to vision model
        self.clip_model.vision_model = get_peft_model(self.clip_model.vision_model, lora_config)

        # ========== 3. Text Branch - CoOp (Context Optimization) ==========
        # Freeze all text encoder parameters
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False

        # Learnable soft prompts (context tokens)
        # Shape: [num_context_tokens, embed_dim]
        # Initialize with random normal distribution (following CoOp paper)
        self.context_tokens = nn.Parameter(
            torch.randn(num_context_tokens, self.embed_dim) * 0.02
        )

        # ========== 4. Build Text Prototypes ==========
        # Text prototypes will be computed from class prompts
        # Shape: [num_classes, embed_dim]
        self.text_prototypes = None
        self.class_names = None

        # Load prompt.json if provided
        if prompt_json_path is not None and os.path.exists(prompt_json_path):
            self.build_text_features(prompt_json_path)

        # Move to device
        self.to(self.device)

    def build_text_features(self, json_path: str) -> torch.Tensor:
        """
        Build text prototypes from prompt.json file.

        For each class:
        1. Read class attributes (contour, pattern) from JSON
        2. Construct hard prompt: "{Class} fragment. [Contour]: {contour} [Pattern]: {pattern}"
        3. Tokenize and get token embeddings
        4. Prepend learnable context tokens
        5. Pass through frozen text encoder
        6. Store as text prototype for this class

        Args:
            json_path (str): Path to prompt.json file

        Returns:
            text_prototypes (torch.Tensor): [num_classes, embed_dim]
        """
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        # Extract class names and attributes
        self.class_names = list(prompt_data.keys())

        # Ensure num_classes matches
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Number of classes in JSON ({len(self.class_names)}) "
                f"does not match num_classes ({self.num_classes})"
            )

        # Build text embeddings for each class
        text_features_list = []

        for class_name in self.class_names:
            class_info = prompt_data[class_name]

            # Construct hard prompt
            # Format: "{Class} fragment. [Contour]: {contour_text} [Pattern]: {pattern_text}"
            contour_text = class_info.get('contour', '')
            pattern_text = class_info.get('pattern', '')

            hard_prompt = f"{class_name} fragment. [Contour]: {contour_text} [Pattern]: {pattern_text}"

            # Tokenize
            inputs = self.tokenizer(
                hard_prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77  # CLIP's standard max sequence length
            )

            # Get token embeddings from CLIP text model
            input_ids = inputs['input_ids'].to(self.device)

            with torch.no_grad():
                # Get token embeddings (before position embedding)
                token_embeds = self.clip_model.text_model.get_input_embeddings()(input_ids)

            # ========== Prepend Context Tokens (CoOp) ==========
            # token_embeds shape: [1, seq_len, embed_dim]
            # context_tokens shape: [num_context_tokens, embed_dim]

            seq_len = token_embeds.shape[1]
            max_seq_len = 77 - self.num_context_tokens  # Reserve space for context tokens

            if seq_len > max_seq_len:
                # Truncate if too long
                token_embeds = token_embeds[:, :max_seq_len, :]

            # Concatenate: [context_tokens] + [token_embeds]
            # Note: We need to handle the special tokens (SOS, EOS) carefully
            # CLIP uses [SOT] (start of text) at position 0 and [EOT] (end of text) at the end

            # Get context tokens as [1, num_context_tokens, embed_dim]
            ctx_tokens = self.context_tokens.unsqueeze(0)  # [1, num_ctx, embed_dim]

            # For CoOp: Replace the first num_context_tokens positions after SOS
            # Standard approach: [SOS, ctx_1, ctx_2, ..., ctx_M, word_tokens, EOS]

            # Get SOS token embedding (position 0)
            sos_embed = token_embeds[:, 0:1, :]  # [1, 1, embed_dim]

            # Get remaining word tokens (excluding SOS)
            word_embeds = token_embeds[:, 1:, :]  # [1, seq_len-1, embed_dim]

            # Construct new sequence: [SOS, ctx_tokens, word_embeds truncated, EOS]
            available_space = 77 - 1 - self.num_context_tokens - 1  # 77 - SOS - ctx - EOS
            word_embeds_truncated = word_embeds[:, :available_space, :]

            # Get EOS token embedding
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 49407  # CLIP's EOT token
            eos_embed = self.clip_model.text_model.get_input_embeddings()(
                torch.tensor([[eos_token_id]], device=self.device)
            )  # [1, 1, embed_dim]

            # Concatenate: SOS + context + words + EOS
            combined_embeds = torch.cat([
                sos_embed,      # [1, 1, embed_dim]
                ctx_tokens,     # [1, num_ctx, embed_dim]
                word_embeds_truncated,  # [1, L, embed_dim]
                eos_embed       # [1, 1, embed_dim]
            ], dim=1)  # [1, total_len, embed_dim]

            # Create position ids (CLIP uses learned position embeddings)
            position_ids = torch.arange(combined_embeds.shape[1], device=self.device).unsqueeze(0)

            # Pass through text encoder
            with torch.no_grad():
                text_outputs = self.clip_model.text_model(
                    inputs_embeds=combined_embeds,
                    position_ids=position_ids,
                    return_dict=True
                )

                # Get the final representation (pooler_output or last_hidden_state[:, 0])
                # CLIP uses the EOT token embedding at the end
                text_feature = text_outputs.pooler_output  # [1, embed_dim]

                # Normalize
                text_feature = F.normalize(text_feature, p=2, dim=-1)

            text_features_list.append(text_feature)

        # Stack all class features
        self.text_prototypes = torch.cat(text_features_list, dim=0)  # [num_classes, embed_dim]

        return self.text_prototypes

    def build_simple_text_features(self, class_names: List[str]) -> torch.Tensor:
        """
        Build text prototypes using simple class names (without JSON attributes).

        Args:
            class_names (list): List of class names

        Returns:
            text_prototypes (torch.Tensor): [num_classes, embed_dim]
        """
        self.class_names = class_names

        text_features_list = []

        for class_name in class_names:
            # Simple prompt: "a photo of {class_name}"
            prompt = f"a photo of {class_name}"

            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            )

            input_ids = inputs['input_ids'].to(self.device)

            with torch.no_grad():
                token_embeds = self.clip_model.text_model.get_input_embeddings()(input_ids)

            # Prepend context tokens
            ctx_tokens = self.context_tokens.unsqueeze(0)
            sos_embed = token_embeds[:, 0:1, :]
            word_embeds = token_embeds[:, 1:, :]

            available_space = 77 - 1 - self.num_context_tokens - 1
            word_embeds_truncated = word_embeds[:, :available_space, :]

            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 49407
            eos_embed = self.clip_model.text_model.get_input_embeddings()(
                torch.tensor([[eos_token_id]], device=self.device)
            )

            combined_embeds = torch.cat([sos_embed, ctx_tokens, word_embeds_truncated, eos_embed], dim=1)
            position_ids = torch.arange(combined_embeds.shape[1], device=self.device).unsqueeze(0)

            with torch.no_grad():
                text_outputs = self.clip_model.text_model(
                    inputs_embeds=combined_embeds,
                    position_ids=position_ids,
                    return_dict=True
                )
                text_feature = F.normalize(text_outputs.pooler_output, p=2, dim=-1)

            text_features_list.append(text_feature)

        self.text_prototypes = torch.cat(text_features_list, dim=0)
        return self.text_prototypes

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal classification.

        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            return_features (bool): If True, also return visual features for prototype update

        Returns:
            dict with keys:
                - 'logits': Classification logits [B, num_classes] (cosine_sim / temperature)
                - 'feat': Visual features [B, embed_dim] (normalized)
                - 'proj_feat': Same as feat for compatibility with existing algorithms
        """
        # Ensure text prototypes are built
        if self.text_prototypes is None:
            raise ValueError("Text prototypes not built. Call build_text_features() or build_simple_text_features() first.")

        # ========== Visual Feature Extraction with LoRA ==========
        # Preprocess images for CLIP
        # CLIP expects images normalized with specific mean/std
        # The processor handles this, but we can also do it manually

        # Get visual features through LoRA-adapted vision encoder
        vision_outputs = self.clip_model.vision_model(pixel_values=x, return_dict=True)

        # Get pooled visual features
        # CLIP vision model has pooler_output (after attention pooling)
        visual_features = vision_outputs.pooler_output  # [B, embed_dim]

        # Normalize visual features
        visual_features_norm = F.normalize(visual_features, p=2, dim=-1)

        # ========== Compute Cosine Similarity with Text Prototypes ==========
        # visual_features_norm: [B, embed_dim]
        # text_prototypes: [num_classes, embed_dim] (already normalized)

        # Compute similarity
        cosine_sim = torch.mm(visual_features_norm, self.text_prototypes.T)  # [B, num_classes]

        # Scale by temperature
        logits = cosine_sim / self.temperature

        # ========== Return Results ==========
        result_dict = {
            'logits': logits,
            'feat': visual_features_norm,
            'proj_feat': visual_features_norm  # For compatibility with OBF-MANet
        }

        return result_dict

    def get_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features only (for prototype update in training).

        Args:
            x (torch.Tensor): Input images [B, C, H, W]

        Returns:
            visual_features (torch.Tensor): [B, embed_dim] normalized
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=x, return_dict=True)
        visual_features = vision_outputs.pooler_output
        return F.normalize(visual_features, p=2, dim=-1)

    def update_text_prototypes(self) -> None:
        """
        Re-compute text prototypes with current context_tokens.
        Call this after context_tokens are updated during training.
        """
        if self.class_names is None:
            return

        # Re-build text features with current context_tokens
        # This will use the updated learnable soft prompts
        if hasattr(self, '_prompt_json_path') and self._prompt_json_path is not None:
            self.build_text_features(self._prompt_json_path)
        else:
            self.build_simple_text_features(self.class_names)

    def no_weight_decay(self):
        """
        Return parameters that should not use weight decay.
        Context tokens typically don't use weight decay in CoOp.
        """
        return {'context_tokens'}

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable parameters (LoRA + context tokens).

        Returns:
            dict of parameter names and tensors
        """
        trainable_params = {}

        # LoRA parameters from vision encoder
        for name, param in self.clip_model.vision_model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param

        # Context tokens
        trainable_params['context_tokens'] = self.context_tokens

        return trainable_params

    def print_trainable_parameters(self) -> None:
        """
        Print information about trainable parameters.
        """
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable parameters: {trainable_params} / {all_params} = {100 * trainable_params / all_params:.2f}%")

    def extra_repr(self) -> str:
        """
        Extra representation for print(model).
        """
        return (
            f"num_classes={self.num_classes}, "
            f"num_context_tokens={self.num_context_tokens}, "
            f"temperature={self.temperature}, "
            f"pretrained_model={self.pretrained_model_name}"
        )


# ========== Factory Function for USB Compatibility ==========
def clip_lora_coop(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_context_tokens: int = 4,
    temperature: float = 0.07,
    **kwargs
) -> CLIPLoRACoOp:
    """
    Factory function to create CLIPLoRACoOp model.

    This follows USB's model factory pattern for compatibility with
    get_net_builder() and config files.

    Args:
        num_classes (int): Number of classes
        pretrained (bool): Whether to load pretrained CLIP (always True for this model)
        pretrained_path (str): Not used (CLIP loads from HuggingFace)
        prompt_json_path (str): Path to prompt.json for class attributes
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout
        num_context_tokens (int): Number of CoOp context tokens
        temperature (float): Temperature for similarity scaling

    Returns:
        CLIPLoRACoOp model instance
    """
    model = CLIPLoRACoOp(
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_context_tokens=num_context_tokens,
        temperature=temperature,
        prompt_json_path=prompt_json_path,
        **kwargs
    )

    return model


def clip_lora_coop_small(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """CLIPLoRACoOp with small LoRA rank (r=4)."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        lora_r=4,
        lora_alpha=8,
        **kwargs
    )


def clip_lora_coop_large(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """CLIPLoRACoOp with large LoRA rank (r=16)."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        lora_r=16,
        lora_alpha=32,
        **kwargs
    )