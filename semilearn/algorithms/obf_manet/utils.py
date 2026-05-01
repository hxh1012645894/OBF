# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from copy import deepcopy
from collections import Counter

from semilearn.algorithms.hooks import MaskingHook


class OBFMANetThresholdingHook(MaskingHook):
    """
    Adaptive Thresholding Hook for OBF-MANet.
    Inherits FlexMatch's adaptive class-wise thresholding mechanism.
    """
    def __init__(self, ulb_dest_len, num_classes, thresh_warmup=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ulb_dest_len = ulb_dest_len
        self.num_classes = num_classes
        self.thresh_warmup = thresh_warmup
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs):
        """Update class-wise accuracy based on selected pseudo-labels."""
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, idx_ulb, softmax_x_ulb=True, *args, **kwargs):
        """
        Compute adaptive mask based on FlexMatch's curriculum pseudo-labeling.

        Returns:
            mask: binary mask for high-confidence samples
            max_idx: predicted class indices
            max_probs: confidence scores
        """
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = self.compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()

        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)

        # FlexMatch convex thresholding: p_cutoff * (class_acc / (2 - class_acc))
        mask = max_probs.ge(algorithm.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))
        select = max_probs.ge(algorithm.p_cutoff)
        mask = mask.to(max_probs.dtype)

        # Update selected labels for samples above base threshold
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()

        return mask