# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import math

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from .utils import OBFMANetThresholdingHook


@ALGORITHMS.register('obf_manet')
class OBFMANet(AlgorithmBase):
    """
    OBF-MANet: Momentum Prototype Consistency with Adaptive Negative Learning

    Based on FlexMatch with two core improvements:
    1. Momentum Prototype Consistency (MPC): Uses EMA-updated class prototypes
       for contrastive learning via InfoNCE loss
    2. Adaptive Negative Learning (ANL): Pushes features away from non-target
       class prototypes for high-confidence samples

    Args:
        T_p (float): Temperature for MPC InfoNCE loss (default: 0.1)
        T_n (float): Temperature for ANL negative learning (default: 0.5)
        momentum (float): EMA momentum for prototype update (default: 0.999)
        gamma (float): Entropy filtering threshold coefficient (default: 0.7)
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(
            T=args.T,
            p_cutoff=args.p_cutoff,
            hard_label=args.hard_label,
            thresh_warmup=args.thresh_warmup,
            T_p=args.T_p,
            T_n=args.T_n,
            proto_momentum=args.proto_momentum,
            gamma=args.gamma
        )

    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True,
             T_p=0.1, T_n=0.5, proto_momentum=0.999, gamma=0.7):
        """Initialize hyperparameters and prototype buffers."""
        # FlexMatch base parameters
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

        # OBF-MANet specific parameters
        self.T_p = T_p
        self.T_n = T_n
        self.proto_momentum = proto_momentum
        self.gamma = gamma

        # Global momentum prototypes [num_classes, 64]
        self.prototypes = torch.zeros(self.num_classes, 64)

        # Prototype initialization status [num_classes]
        self.prototype_initialized = torch.zeros(self.num_classes, dtype=torch.bool)

    def set_hooks(self):
        """Register necessary hooks for pseudo-labeling and thresholding."""
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            OBFMANetThresholdingHook(
                ulb_dest_len=self.args.ulb_dest_len,
                num_classes=self.num_classes,
                thresh_warmup=self.args.thresh_warmup
            ),
            "MaskingHook"
        )
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        Core training logic for OBF-MANet.

        Flow:
        1. Initialize prototypes with labeled data (first encounter per class)
        2. Compute supervised loss
        3. FlexMatch pseudo-label generation with adaptive thresholding
        4. Compute unsupervised consistency loss
        5. Momentum prototype update (MPC core)
        6. Compute MPC Loss (InfoNCE)
        7. Compute ANL Loss (adaptive negative learning)
        8. Aggregate total loss
        """
        num_lb = y_lb.shape[0]
        device = x_lb.device

        with self.amp_cm():
            # ========== Forward Pass ==========
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
                proj_x_lb = outputs['proj_feat'][:num_lb]
                proj_x_ulb_w, proj_x_ulb_s = outputs['proj_feat'][num_lb:].chunk(2)
            else:
                # Labeled data
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                proj_x_lb = outs_x_lb['proj_feat']

                # Strong augmented unlabeled data
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                proj_x_ulb_s = outs_x_ulb_s['proj_feat']

                # Weak augmented unlabeled data (no_grad)
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
                    proj_x_ulb_w = outs_x_ulb_w['proj_feat']

            feat_dict = {
                'x_lb': feats_x_lb,
                'x_ulb_w': feats_x_ulb_w,
                'x_ulb_s': feats_x_ulb_s
            }

            # ========== 1. Initialize Prototypes with Labeled Data ==========
            # Move prototypes to correct device
            if self.prototypes.device != device:
                self.prototypes = self.prototypes.to(device)
                self.prototype_initialized = self.prototype_initialized.to(device)

            self._init_prototypes_with_labels(proj_x_lb, y_lb)

            # ========== 2. Supervised Loss ==========
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # ========== 3. FlexMatch Pseudo-labels and Threshold Mask ==========
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook",
                                                probs_x_ulb=probs_x_ulb_w.detach())

            mask = self.call_hook("masking", "MaskingHook",
                                  logits_x_ulb=probs_x_ulb_w,
                                  softmax_x_ulb=False,
                                  idx_ulb=idx_ulb)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # ========== 4. Unsupervised Consistency Loss ==========
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            # ========== 5-7. MPC & ANL Losses (require prototype initialization) ==========
            loss_mpc = torch.tensor(0.0, device=device)
            loss_anl = torch.tensor(0.0, device=device)

            if self.prototype_initialized.all():
                # 5. Momentum prototype update
                self._update_prototypes(proj_x_ulb_w, probs_x_ulb_w, mask)

                # 6. Compute MPC Loss (InfoNCE)
                loss_mpc = self._compute_mpc_loss(proj_x_ulb_s, pseudo_label, mask)

                # 7. Compute ANL Loss
                loss_anl = self._compute_anl_loss(proj_x_ulb_w, probs_x_ulb_w)

            # ========== 8. Aggregate Total Loss ==========
            total_loss = sup_loss + self.lambda_u * unsup_loss + loss_mpc + loss_anl

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            mpc_loss=loss_mpc.item(),
            anl_loss=loss_anl.item(),
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item(),
            proto_init_ratio=self.prototype_initialized.float().mean().item()
        )
        return out_dict, log_dict

    # ========== Core Helper Methods ==========

    @torch.no_grad()
    def _init_prototypes_with_labels(self, proj_features_x, labels_x):
        """
        Initialize class prototypes using labeled data.
        Only initializes prototypes for classes that haven't been initialized yet.

        Args:
            proj_features_x: projected features from labeled samples [B, 64]
            labels_x: ground-truth labels [B]
        """
        # Find uninitialized classes
        uninitialized_classes = (~self.prototype_initialized).nonzero(as_tuple=True)[0]

        if len(uninitialized_classes) == 0:
            return  # All classes already initialized

        # Check which uninitialized classes appear in current batch
        unique_labels = labels_x.unique()

        for c in uninitialized_classes:
            if c in unique_labels:
                # Get all samples belonging to class c
                class_mask = (labels_x == c)
                class_features = proj_features_x[class_mask]

                # Compute mean feature for this class
                feature_mean = class_features.mean(dim=0)

                # L2 normalize the prototype
                prototype = F.normalize(feature_mean, p=2, dim=0)

                # Assign to prototypes buffer
                self.prototypes[c] = prototype

                # Mark as initialized
                self.prototype_initialized[c] = True

    @torch.no_grad()
    def _update_prototypes(self, proj_features_ulb_w, probs_ulb_w, mask):
        """
        Update momentum prototypes using EMA from high-confidence unlabeled samples.

        Only updates prototypes for samples whose confidence exceeds the FlexMatch
        dynamic threshold (mask == 1).

        Args:
            proj_features_ulb_w: projected features from weak augmented unlabeled data [B_ulb, 64]
            probs_ulb_w: probability predictions from weak augmented unlabeled data [B_ulb, num_classes]
            mask: binary mask for high-confidence samples [B_ulb]
        """
        # Get high-confidence samples
        high_conf_mask = (mask > 0.5)
        if high_conf_mask.sum() == 0:
            return

        # Get pseudo-labels (argmax) for high-confidence samples
        pseudo_labels = probs_ulb_w[high_conf_mask].argmax(dim=-1)
        high_conf_features = proj_features_ulb_w[high_conf_mask]

        # Group features by pseudo-label and update prototypes
        unique_labels = pseudo_labels.unique()

        for c in unique_labels:
            if not self.prototype_initialized[c]:
                continue  # Skip uninitialized classes

            class_mask = (pseudo_labels == c)
            if class_mask.sum() == 0:
                continue

            # Compute mean feature for this pseudo-class
            class_features = high_conf_features[class_mask]
            feature_mean = class_features.mean(dim=0)

            # L2 normalize
            feature_mean = F.normalize(feature_mean, p=2, dim=0)

            # EMA update: prototype = proto_momentum * prototype + (1 - proto_momentum) * feature_mean
            self.prototypes[c] = F.normalize(
                self.proto_momentum * self.prototypes[c] + (1 - self.proto_momentum) * feature_mean,
                p=2, dim=0
            )

    def _compute_mpc_loss(self, proj_features_ulb_s, pseudo_label, mask):
        """
        Compute MPC Loss using InfoNCE contrastive formulation.

        For each high-confidence sample:
        - Positive: its pseudo-label's prototype
        - Negatives: all other class prototypes

        Args:
            proj_features_ulb_s: projected features from strong augmented unlabeled data [B_ulb, 64]
            pseudo_label: pseudo-labels (hard or soft) [B_ulb] or [B_ulb, num_classes]
            mask: binary mask for high-confidence samples [B_ulb]

        Returns:
            InfoNCE loss for prototype alignment
        """
        high_conf_mask = (mask > 0.5)
        if high_conf_mask.sum() == 0:
            return torch.tensor(0.0, device=proj_features_ulb_s.device)

        # Get high-confidence features and pseudo-labels
        features = proj_features_ulb_s[high_conf_mask]

        # Handle hard vs soft pseudo-labels
        if pseudo_label.dim() == 2:
            # Soft labels: use argmax
            targets = pseudo_label[high_conf_mask].argmax(dim=-1)
        else:
            targets = pseudo_label[high_conf_mask]

        # Compute similarity with all prototypes [B, num_classes]
        # prototypes are already L2 normalized, features need normalization
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = self.prototypes  # already normalized

        similarity = torch.mm(features_norm, prototypes_norm.T) / self.T_p

        # InfoNCE loss: cross entropy with pseudo-label as target
        loss_mpc = F.cross_entropy(similarity, targets, reduction='mean')

        return loss_mpc

    def _compute_anl_loss(self, proj_features_ulb_w, probs_ulb_w):
        """
        Compute Adaptive Negative Learning Loss.

        For samples with low entropy (high reliability):
        - Push features away from non-target class prototypes
        - Use softmax over non-target similarities to weight the negative targets

        Args:
            proj_features_ulb_w: projected features from weak augmented unlabeled data [B_ulb, 64]
            probs_ulb_w: probability predictions from weak augmented unlabeled data [B_ulb, num_classes]

        Returns:
            ANL loss for negative learning
        """
        # Compute entropy H(p) for each sample
        entropy = -torch.sum(probs_ulb_w * torch.log(probs_ulb_w + 1e-8), dim=1)

        # Entropy threshold: gamma * log(num_classes)
        entropy_threshold = self.gamma * math.log(self.num_classes)

        # Reliability mask: only keep samples with entropy < threshold
        reliable_mask = (entropy < entropy_threshold)
        if reliable_mask.sum() == 0:
            return torch.tensor(0.0, device=proj_features_ulb_w.device)

        # Get reliable features and their predictions
        reliable_features = proj_features_ulb_w[reliable_mask]
        reliable_probs = probs_ulb_w[reliable_mask]

        # Get predicted class (target class)
        target_classes = reliable_probs.argmax(dim=-1)

        # Normalize features
        features_norm = F.normalize(reliable_features, p=2, dim=1)
        prototypes_norm = self.prototypes  # already normalized

        # Compute similarity with all prototypes [B_reliable, num_classes]
        similarity = torch.mm(features_norm, prototypes_norm.T)

        # Create mask for non-target classes
        batch_size = reliable_features.shape[0]
        non_target_mask = torch.ones(batch_size, self.num_classes, device=features_norm.device)
        non_target_mask.scatter_(1, target_classes.unsqueeze(1), 0)

        # Compute negative weights via softmax over non-target similarities
        # Apply temperature T_n
        negative_similarity = similarity.clone()
        negative_similarity[~non_target_mask.bool()] = float('-inf')  # mask target class
        negative_weights = F.softmax(negative_similarity / self.T_n, dim=1)

        # Gaussian warmup coefficient lambda(t)
        # lambda(t) = 1 - exp(-t / max_iter) approximated by linear warmup
        warmup_coef = min(1.0, self.it / (self.num_train_iter * 0.5))

        # ANL loss: weighted cross entropy pushing away from non-target prototypes
        # Target: all zeros (push away from all non-target classes)
        # We use negative log of (1 - softmax_prob_for_non_target) as the loss

        # For each sample, we want to minimize similarity with non-target prototypes
        # Loss = -sum(weights * log(1 - softmax_prob))
        # But a simpler formulation: cross entropy with uniform distribution over non-targets

        # Alternative: push features away by maximizing distance = minimizing similarity
        # Use cross entropy where we want the model to predict "no class" for non-targets

        # Simpler formulation: weighted sum of similarities to non-targets
        # Loss = sum_i sum_{j != target_i} w_ij * similarity_ij
        # where w_ij = softmax over non-target similarities

        weighted_non_target_sim = (negative_weights * similarity * non_target_mask).sum(dim=1)

        # We want to minimize this, so take mean
        # Apply warmup coefficient
        loss_anl = warmup_coef * weighted_non_target_sim.mean()

        return loss_anl

    def get_save_dict(self):
        """Save prototypes and initialization status along with base model state."""
        save_dict = super().get_save_dict()
        save_dict['prototypes'] = self.prototypes.cpu()
        save_dict['prototype_initialized'] = self.prototype_initialized.cpu()
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        """Load prototypes and initialization status from checkpoint."""
        checkpoint = super().load_model(load_path)

        if 'prototypes' in checkpoint:
            self.prototypes = checkpoint['prototypes'].cuda(self.gpu)
        if 'prototype_initialized' in checkpoint:
            self.prototype_initialized = checkpoint['prototype_initialized'].cuda(self.gpu)
        if 'classwise_acc' in checkpoint:
            self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        if 'selected_label' in checkpoint:
            self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)

        self.print_fn("OBF-MANet additional parameters loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        """Return CLI arguments for OBF-MANet."""
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
            SSL_Argument('--T_p', float, 0.1),
            SSL_Argument('--T_n', float, 0.5),
            SSL_Argument('--proto_momentum', float, 0.999),
            SSL_Argument('--gamma', float, 0.7),
        ]