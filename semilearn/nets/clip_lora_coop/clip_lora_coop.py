# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# OBF-MANet-V2: Multi-modal Semi-supervised Learning with CLIP + LoRA + CoOp
# Refactored with ablation experiment mode switches

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
    CLIP with configurable training modes for ablation experiments.

    Supported modes:
    1. Full mode (default): LoRA on vision + CoOp on text + multimodal classification
    2. Zero-shot mode: Frozen CLIP with text prototypes (use_lora=False, use_coop=False)
    3. Linear probe mode: Frozen vision + linear classifier (use_linear_probe=True)
    4. Full fine-tuning mode: Unfreeze vision backbone (freeze_vision=False)
    5. CoOp-only mode: No LoRA, only soft prompts (use_lora=False, use_coop=True)
    6. LoRA-only mode: No CoOp, only LoRA (use_lora=True, use_coop=False)

    Args:
        num_classes (int): Number of classes for classification
        use_lora (bool): Whether to use LoRA on vision encoder (default: True)
        use_coop (bool): Whether to use learnable soft prompts (default: True)
        freeze_vision (bool): Whether to freeze vision backbone (default: True)
        use_linear_probe (bool): Whether to use linear classifier instead of multimodal (default: False)
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
        use_lora: bool = True,
        use_coop: bool = True,
        freeze_vision: bool = True,
        use_linear_probe: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_context_tokens: int = 4,
        temperature: float = 0.07,
        pretrained_model_name: str = 'openai/clip-vit-base-patch16',
        prompt_json_path: Optional[str] = None,
        proj_size: Optional[int] = None,  # For OBF-MANet contrastive learning
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        # Store mode switches
        self.use_lora = use_lora
        self.use_coop = use_coop
        self.freeze_vision = freeze_vision
        self.use_linear_probe = use_linear_probe
        self.proj_size = proj_size  # 64 for OBF-MANet, None for no projection

        self.num_classes = num_classes
        self.num_context_tokens = num_context_tokens
        self.temperature = temperature
        self.pretrained_model_name = pretrained_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ========== 1. Load Pretrained CLIP Model ==========
        # Use safetensors to avoid torch.load security vulnerability (requires torch>=2.6)
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name)

        # Get embedding dimension from CLIP config
        self.embed_dim = self.clip_model.config.text_config.hidden_size  # Typically 512 for clip-vit-base
        self.vision_hidden_size = self.clip_model.config.vision_config.hidden_size  # Typically 768 for clip-vit-base

        # ========== 2. Visual Branch - Configurable Fine-tuning ==========

        # Freeze/Unfreeze vision backbone
        if self.freeze_vision:
            # Freeze all visual encoder parameters
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        else:
            # Unfreeze all visual encoder parameters for full fine-tuning
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = True

        # LoRA: Only apply when use_lora=True AND freeze_vision=True
        # (LoRA is meant for efficient fine-tuning when backbone is frozen)
        if self.use_lora and self.freeze_vision:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['q_proj', 'v_proj'],  # Target Query and Value projections
                modules_to_save=None,
            )
            # Apply LoRA to vision model
            self.clip_model.vision_model = get_peft_model(self.clip_model.vision_model, lora_config)

        # Linear probe classifier: Only when use_linear_probe=True
        if self.use_linear_probe:
            # Use vision_hidden_size for linear classifier input
            self.classifier = nn.Linear(self.vision_hidden_size, num_classes)

        # Projection head for OBF-MANet contrastive learning
        # Projects vision_hidden_size (768) to proj_size (64)
        if self.proj_size is not None:
            self.projector = nn.Sequential(
                nn.Linear(self.vision_hidden_size, self.vision_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.vision_hidden_size // 2, self.proj_size)
            )
            print(f"[CLIP] Added projection head: {self.vision_hidden_size} -> {self.proj_size}")
        else:
            self.projector = None

        # ========== 3. Text Branch - CoOp (Context Optimization) ==========
        # Freeze all text encoder parameters (always frozen)
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False

        # Learnable soft prompts (context tokens) - only when use_coop=True
        if self.use_coop:
            # Shape: [num_context_tokens, embed_dim]
            # Initialize with random normal distribution (following CoOp paper)
            self.context_tokens = nn.Parameter(
                torch.randn(num_context_tokens, self.embed_dim) * 0.02
            )
        else:
            # No learnable context tokens
            self.context_tokens = None

        # ========== 4. Build Text Prototypes ==========
        # Text prototypes will be computed from class prompts
        # Shape: [num_classes, embed_dim]
        self.text_prototypes = None
        self.class_names = None

        # Move to device BEFORE building text features (so embeddings are on correct device)
        self.to(self.device)

        # Load prompt.json if provided, otherwise build default text features
        if prompt_json_path is not None and os.path.exists(prompt_json_path):
            self.build_text_features(prompt_json_path)
        elif not self.use_linear_probe:
            # No prompt.json provided: build default text features using generic class names
            # This is needed for multimodal classification mode
            default_class_names = [f"class_{i}" for i in range(num_classes)]
            self.build_simple_text_features(default_class_names)
            print(f"[CLIP] Built default text features for {num_classes} classes (no prompt.json provided)")

        # Print configuration info
        self._print_mode_info()

    def _print_mode_info(self) -> None:
        """Print current mode configuration."""
        mode_name = self._get_mode_name()
        print(f"[CLIPLoRACoOp] Mode: {mode_name}")
        print(f"  - use_lora: {self.use_lora}")
        print(f"  - use_coop: {self.use_coop}")
        print(f"  - freeze_vision: {self.freeze_vision}")
        print(f"  - use_linear_probe: {self.use_linear_probe}")

    def _get_mode_name(self) -> str:
        """Get human-readable mode name."""
        if self.use_linear_probe:
            return "Linear Probe"
        elif not self.freeze_vision:
            return "Full Fine-tuning"
        elif self.use_lora and self.use_coop:
            return "LoRA + CoOp (Full)"
        elif self.use_lora and not self.use_coop:
            return "LoRA-only"
        elif not self.use_lora and self.use_coop:
            return "CoOp-only"
        else:
            return "Zero-shot CLIP"

    def build_text_features(self, json_path: str) -> torch.Tensor:
        """
        Build text prototypes from prompt.json file.

        JSON format (natural language):
        {
            "01": {
                "class_name": "Left Epiplastron",
                "description": "A close-up photo of..."
            },
            ...
        }

        For each class:
        1. Read description from JSON
        2. Tokenize and get token embeddings
        3. Prepend learnable context tokens (CoOp) if use_coop=True
        4. Pass through frozen text encoder
        5. L2 normalize and store as text prototype

        Args:
            json_path (str): Path to prompt.json file

        Returns:
            text_prototypes (torch.Tensor): [num_classes, embed_dim]
        """
        # Store path for potential re-computation
        self._prompt_json_path = json_path

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        # Sort by numeric key and extract class names
        sorted_keys = sorted(prompt_data.keys(), key=lambda x: int(x))
        self.class_names = [prompt_data[k].get('class_name', k) for k in sorted_keys]

        # Ensure num_classes matches
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Number of classes in JSON ({len(self.class_names)}) "
                f"does not match num_classes ({self.num_classes})"
            )

        # Build text embeddings for each class
        text_features_list = []

        for key in sorted_keys:
            class_info = prompt_data[key]

            # ========== Use description field (natural format) ==========
            if 'description' in class_info:
                hard_prompt = class_info['description']
            else:
                # Fallback to class name
                class_name = class_info.get('class_name', key)
                hard_prompt = f"a photo of {class_name}"

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
                # Directly access token embedding layer (compatible with all transformers versions)
                token_embeds = self.clip_model.text_model.embeddings.token_embedding(input_ids)

            # ========== Prepend Context Tokens (CoOp) - Conditional ==========
            if self.use_coop and self.context_tokens is not None:
                # With CoOp: prepend learnable soft prompts
                # [SOS, ctx_1, ctx_2, ..., ctx_M, word_tokens, EOS]

                ctx_tokens = self.context_tokens.unsqueeze(0)  # [1, num_ctx, embed_dim]
                sos_embed = token_embeds[:, 0:1, :]  # [1, 1, embed_dim]
                word_embeds = token_embeds[:, 1:, :]  # [1, seq_len-1, embed_dim]

                # Calculate available space: 77 = SOS(1) + Context(M) + Words(L) + EOS(1)
                available_space = 77 - 1 - self.num_context_tokens - 1
                word_embeds_truncated = word_embeds[:, :available_space, :]

                eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 49407
                eos_embed = self.clip_model.text_model.embeddings.token_embedding(
                    torch.tensor([[eos_token_id]], device=self.device)
                )

                combined_embeds = torch.cat([
                    sos_embed,
                    ctx_tokens,
                    word_embeds_truncated,
                    eos_embed
                ], dim=1)

            else:
                # Without CoOp: use pure hard prompt tokens
                # Just use the original token embeddings (already includes SOS and EOS)
                combined_embeds = token_embeds

            # Create position ids
            position_ids = torch.arange(combined_embeds.shape[1], device=self.device).unsqueeze(0)

            # Pass through text encoder (handle different transformers versions)
            with torch.no_grad():
                # Try modern API first, fall back to manual layer-by-layer if needed
                try:
                    text_outputs = self.clip_model.text_model(
                        inputs_embeds=combined_embeds,
                        position_ids=position_ids,
                        return_dict=True
                    )
                    text_feature = text_outputs.pooler_output
                except TypeError:
                    # Fallback for older transformers: manual forward pass
                    text_feature = self._forward_text_encoder_legacy(combined_embeds, position_ids)

                # L2 normalize
                text_feature = F.normalize(text_feature, p=2, dim=-1)

            text_features_list.append(text_feature)

        # Stack all class features
        self.text_prototypes = torch.cat(text_features_list, dim=0)  # [num_classes, embed_dim]

        return self.text_prototypes

    def _forward_text_encoder_legacy(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Manual forward pass for older transformers versions that don't support inputs_embeds.

        Args:
            inputs_embeds: Token embeddings [1, seq_len, embed_dim]
            position_ids: Position IDs [1, seq_len]

        Returns:
            text_feature: Pooled text feature [1, embed_dim]
        """
        seq_len = inputs_embeds.shape[1]
        bsz = inputs_embeds.shape[0]

        # Get position embeddings
        position_embeds = self.clip_model.text_model.embeddings.position_embedding(position_ids)

        # Combine token + position embeddings
        hidden_states = inputs_embeds + position_embeds

        # Create attention masks in correct format for CLIP
        # attention_mask: [bsz, 1, tgt_len, src_len] - all ones for valid tokens
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), device=self.device)

        # causal_attention_mask: [bsz, 1, tgt_len, src_len] - causal mask
        causal_attention_mask = self._create_causal_mask(seq_len, bsz, self.device)

        # Pass through encoder layers
        for encoder_layer in self.clip_model.text_model.encoder.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask
            )

        # Apply final layer norm
        hidden_states = self.clip_model.text_model.final_layer_norm(hidden_states)

        # Get EOS token representation (last token)
        text_feature = hidden_states[:, -1, :]

        return text_feature

    def _create_causal_mask(self, seq_len: int, bsz: int, device: str) -> torch.Tensor:
        """
        Create causal attention mask for CLIP text encoder.

        Args:
            seq_len: Sequence length
            bsz: Batch size
            device: Device to create mask on

        Returns:
            causal_mask: [bsz, 1, seq_len, seq_len]
        """
        # Create lower triangular mask (causal) - 0 for valid, -inf for invalid
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)  # [bsz, 1, seq_len, seq_len]

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
                token_embeds = self.clip_model.text_model.embeddings.token_embedding(input_ids)

            # Prepend context tokens if use_coop=True
            if self.use_coop and self.context_tokens is not None:
                ctx_tokens = self.context_tokens.unsqueeze(0)
                sos_embed = token_embeds[:, 0:1, :]
                word_embeds = token_embeds[:, 1:, :]

                available_space = 77 - 1 - self.num_context_tokens - 1
                word_embeds_truncated = word_embeds[:, :available_space, :]

                eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 49407
                eos_embed = self.clip_model.text_model.embeddings.token_embedding(
                    torch.tensor([[eos_token_id]], device=self.device)
                )

                combined_embeds = torch.cat([sos_embed, ctx_tokens, word_embeds_truncated, eos_embed], dim=1)
            else:
                combined_embeds = token_embeds

            position_ids = torch.arange(combined_embeds.shape[1], device=self.device).unsqueeze(0)

            with torch.no_grad():
                # Try modern API first, fall back to manual if needed
                try:
                    text_outputs = self.clip_model.text_model(
                        inputs_embeds=combined_embeds,
                        position_ids=position_ids,
                        return_dict=True
                    )
                    text_feature = text_outputs.pooler_output
                except TypeError:
                    # Fallback for older transformers
                    text_feature = self._forward_text_encoder_legacy(combined_embeds, position_ids)

                text_feature = F.normalize(text_feature, p=2, dim=-1)

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
        Forward pass with configurable output routing.

        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            return_features (bool): If True, also return visual features for prototype update

        Returns:
            dict with keys:
                - 'logits': Classification logits [B, num_classes]
                - 'feat': Visual features [B, embed_dim] (normalized)
                - 'proj_feat': Same as feat for compatibility with existing algorithms
        """
        # ========== Visual Feature Extraction ==========
        vision_outputs = self.clip_model.vision_model(pixel_values=x, return_dict=True)
        visual_features = vision_outputs.pooler_output  # [B, vision_hidden_size]

        # Normalize visual features
        visual_features_norm = F.normalize(visual_features, p=2, dim=-1)

        # ========== Output Routing ==========
        if self.use_linear_probe:
            # Linear probe mode: use linear classifier
            logits = self.classifier(visual_features)  # [B, num_classes]
        else:
            # Multimodal mode: compute cosine similarity with text prototypes
            if self.text_prototypes is None:
                raise ValueError(
                    "Text prototypes not built. Call build_text_features() or "
                    "build_simple_text_features() first, or set use_linear_probe=True."
                )

            # visual_features_norm: [B, embed_dim]
            # text_prototypes: [num_classes, embed_dim] (already normalized)
            cosine_sim = torch.mm(visual_features_norm, self.text_prototypes.T)  # [B, num_classes]
            logits = cosine_sim / self.temperature

        # ========== Return Results ==========
        # Compute projection features for OBF-MANet if projector exists
        if self.projector is not None:
            # Use raw visual_features (before normalization) for projection
            proj_features = self.projector(visual_features)  # [B, proj_size]
            proj_features_norm = F.normalize(proj_features, p=2, dim=-1)
        else:
            # No projection, use normalized visual features directly
            proj_features_norm = visual_features_norm

        result_dict = {
            'logits': logits,
            'feat': visual_features_norm,
            'proj_feat': proj_features_norm  # For compatibility with OBF-MANet
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
        Only effective when use_coop=True.
        """
        if not self.use_coop:
            return  # No need to update if CoOp is disabled

        if self.class_names is None:
            return

        if hasattr(self, '_prompt_json_path') and self._prompt_json_path is not None:
            self.build_text_features(self._prompt_json_path)
        else:
            self.build_simple_text_features(self.class_names)

    def no_weight_decay(self):
        """
        Return parameters that should not use weight decay.
        Context tokens typically don't use weight decay in CoOp.
        """
        if self.use_coop:
            return {'context_tokens'}
        return set()

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable parameters based on current mode.

        Returns:
            dict of parameter names and tensors
        """
        trainable_params = {}

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param

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
        mode_name = self._get_mode_name()
        return (
            f"mode={mode_name}, "
            f"num_classes={self.num_classes}, "
            f"use_lora={self.use_lora}, "
            f"use_coop={self.use_coop}, "
            f"freeze_vision={self.freeze_vision}, "
            f"use_linear_probe={self.use_linear_probe}, "
            f"num_context_tokens={self.num_context_tokens if self.use_coop else 'N/A'}, "
            f"temperature={self.temperature}, "
            f"pretrained_model={self.pretrained_model_name}"
        )


# ========== Factory Function for USB Compatibility ==========
def clip_lora_coop(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    use_lora: bool = True,
    use_coop: bool = True,
    freeze_vision: bool = True,
    use_linear_probe: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_context_tokens: int = 4,
    temperature: float = 0.07,
    proj_size: Optional[int] = None,  # For OBF-MANet contrastive learning
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
        use_lora (bool): Whether to use LoRA on vision encoder
        use_coop (bool): Whether to use learnable soft prompts
        freeze_vision (bool): Whether to freeze vision backbone
        use_linear_probe (bool): Whether to use linear classifier
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout
        num_context_tokens (int): Number of CoOp context tokens
        temperature (float): Temperature for similarity scaling
        proj_size (int): Projection size for OBF-MANet contrastive learning (default: None)

    Returns:
        CLIPLoRACoOp model instance
    """
    model = CLIPLoRACoOp(
        num_classes=num_classes,
        use_lora=use_lora,
        use_coop=use_coop,
        freeze_vision=freeze_vision,
        use_linear_probe=use_linear_probe,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_context_tokens=num_context_tokens,
        temperature=temperature,
        prompt_json_path=prompt_json_path,
        proj_size=proj_size,
        **kwargs
    )

    return model


# ========== Preset Factory Functions for Common Modes ==========

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


def clip_zero_shot(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """Zero-shot CLIP mode: frozen backbone, no LoRA, no CoOp."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        use_lora=False,
        use_coop=False,
        freeze_vision=True,
        use_linear_probe=False,
        **kwargs
    )


def clip_linear_probe(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """Linear probe mode: frozen vision backbone + linear classifier."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        use_lora=False,
        use_coop=False,
        freeze_vision=True,
        use_linear_probe=True,
        **kwargs
    )


def clip_coop_only(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """CoOp-only mode: no LoRA, only learnable soft prompts."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        use_lora=False,
        use_coop=True,
        freeze_vision=True,
        use_linear_probe=False,
        **kwargs
    )


def clip_lora_only(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """LoRA-only mode: LoRA on vision, no CoOp soft prompts."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        use_lora=True,
        use_coop=False,
        freeze_vision=True,
        use_linear_probe=False,
        **kwargs
    )


def clip_full_finetune(
    num_classes: int,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    prompt_json_path: Optional[str] = None,
    **kwargs
) -> CLIPLoRACoOp:
    """Full fine-tuning mode: unfreeze entire vision backbone."""
    return clip_lora_coop(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        prompt_json_path=prompt_json_path,
        use_lora=False,  # LoRA not needed when full fine-tuning
        use_coop=True,
        freeze_vision=False,  # Unfreeze backbone
        use_linear_probe=False,
        **kwargs
    )