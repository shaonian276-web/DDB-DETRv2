"""
reference:
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterion(nn.Module):
    """This class computes the loss for DETR."""

    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses,
                 alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # DN-TOD-lite / Noise-aware 参数
        self.na_min_weight = 0.35
        self.na_iou_power = 0.5

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )

        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes,
            num_classes=self.num_classes + 1
        )[..., :-1]

        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits,
            target,
            self.alpha,
            self.gamma,
            reduction='none'
        )

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        """
        DN-TOD-lite VFL:
        1. 使用 IoU 作为正样本质量分数；
        2. 根据 IoU 生成 noise-aware 权重；
        3. 降低疑似框噪声样本对分类分支的影响。
        """
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        ious, _ = box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )

        ious = torch.diag(ious).detach().clamp(min=0.0, max=1.0)

        src_logits = outputs['pred_logits']

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes,
            num_classes=self.num_classes + 1
        )[..., :-1]

        target_score_o = torch.zeros_like(
            target_classes,
            dtype=src_logits.dtype
        )

        target_score_o[idx] = ious.to(target_score_o.dtype)

        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()

        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits,
            target_score,
            weight=weight,
            reduction='none'
        )

        # ---------------- DN-TOD-lite: classification reweighting ----------------
        with torch.no_grad():
            noise_weight = torch.ones_like(target_classes, dtype=src_logits.dtype)

            pos_weight = ious.pow(self.na_iou_power)
            pos_weight = pos_weight.clamp(min=self.na_min_weight, max=1.0)

            noise_weight[idx] = pos_weight.to(noise_weight.dtype)

        loss = loss.mean(-1) * noise_weight

        loss = loss.sum() * src_logits.shape[1] / num_boxes

        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets],
            device=device
        )

        card_pred = (
            pred_logits.argmax(-1) != pred_logits.shape[-1] - 1
        ).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        DN-TOD-lite box loss:
        1. 根据当前预测框与 GT 的 IoU 判断样本可靠性；
        2. IoU 高的样本权重大；
        3. IoU 低的样本可能存在框偏移噪声，降低其回归监督强度。
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        losses = {}

        with torch.no_grad():
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )

            ious = torch.diag(ious).detach().clamp(min=0.0, max=1.0)

            box_weight = ious.pow(self.na_iou_power)
            box_weight = box_weight.clamp(min=self.na_min_weight, max=1.0)

        loss_bbox = F.l1_loss(
            src_boxes,
            target_boxes,
            reduction='none'
        )

        loss_bbox = loss_bbox.sum(-1) * box_weight

        losses['loss_bbox'] = loss_bbox.sum() / box_weight.sum().clamp(min=1.0)

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        )

        loss_giou = loss_giou * box_weight

        losses['loss_giou'] = loss_giou.sum() / box_weight.sum().clamp(min=1.0)

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )

        src_idx = torch.cat(
            [src for (src, _) in indices]
        )

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )

        tgt_idx = torch.cat(
            [tgt for (_, tgt) in indices]
        )

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if 'aux' not in k
        }

        num_boxes = sum(len(t["labels"]) for t in targets)

        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )

        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(
            num_boxes / get_world_size(),
            min=1
        ).item()

        indices = self.matcher(outputs_without_aux, targets)['indices']

        losses = {}

        for loss in self.losses:
            l_dict = self.get_loss(
                loss,
                outputs,
                targets,
                indices,
                num_boxes
            )

            l_dict = {
                k: l_dict[k] * self.weight_dict[k]
                for k in l_dict
                if k in self.weight_dict
            }

            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)['indices']

                for loss in self.losses:
                    if loss == 'masks':
                        continue

                    kwargs = {}

                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                        **kwargs
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }

                    l_dict = {
                        k + f'_aux_{i}': v
                        for k, v in l_dict.items()
                    }

                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''

            indices = self.get_cdn_matched_indices(
                outputs['dn_meta'],
                targets
            )

            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue

                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        dn_num_boxes,
                        **kwargs
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }

                    l_dict = {
                        k + f'_dn_{i}': v
                        for k, v in l_dict.items()
                    }

                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]

        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []

        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(
                    num_gt,
                    dtype=torch.int64,
                    device=device
                )

                gt_idx = gt_idx.tile(dn_num_group)

                assert len(dn_positive_idx[i]) == len(gt_idx)

                dn_match_indices.append(
                    (dn_positive_idx[i], gt_idx)
                )
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device)
                    )
                )

        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()

    correct = pred.eq(
        target.view(1, -1).expand_as(pred)
    )

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res