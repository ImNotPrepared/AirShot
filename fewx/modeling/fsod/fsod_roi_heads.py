# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from .fsod_fast_rcnn import FsodFastRCNNOutputLayers

import time
from detectron2.structures import Boxes, Instances

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

@ROI_HEADS_REGISTRY.register()
class FsodRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES  # IN_FEATURES: ["res2", "res3", "res4"]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales_res2     = (1.0 / input_shape[self.in_features[0]].stride, )
        pooler_scales_res3     = (1.0 / input_shape[self.in_features[1]].stride, )
        pooler_scales_res4     = (1.0 / input_shape[self.in_features[2]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        # assert len(self.in_features) == 1

        self.pooler = {"res2": ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales_res2,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        ),
        "res3": ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales_res3,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        ),
        "res4": ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales_res4,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            
        )}

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FsodFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler["res4"](features, boxes)
        return self.res5(x)

    def roi_pooling(self, features, boxes):
        box_features={}
        box_features["res2"] = self.pooler["res2"](
            [features["res2"]], boxes
        )
        box_features["res3"] = self.pooler["res3"](
            [features["res3"]], boxes
        )
        box_features["res4"] = self.pooler["res4"](
            [features["res4"]], boxes
        )
        '''ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales_res2, 3, 4
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        ),'''
        #feature_pooled = box_features.mean(dim=[2, 3], keepdim=True)  # pooled to 1x1

        return box_features #feature_pooled

    def forward(self, images, features, support_box_features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images
        
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        """  
            It performs box matching between `proposals` and `targets`, and assigns training labels to the proposals.
                Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        
        del targets
        
        proposal_boxes = [x.proposal_boxes for x in proposals] # [128, 4]

        box_features = self._shared_roi_transform(
            [features["res4"]], proposal_boxes
        ) # (box_features.shape): [128, 2048, 7, 7]

        
        #support_features = self.res5(support_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)
        #self.box_predictor = FsodFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1))

        return pred_class_logits, pred_proposal_deltas, proposals

    @torch.no_grad()
    def eval_with_support(self, images, features, support_proposals_dict, support_box_features_dict):
        """
        eval_with_support(query_images, query_features, support_proposals_dict, support_box_features_dict)
        
        support_proposals_dict[cls_id] = proposals = self.proposal_generator(query_images, support_correlation, None)
        support_box_features_dict[cls_id] = support_box_features = self.support_dict['res5_avg'][cls_id]
        
        
        See :meth:`ROIHeads.forward`.
        """
        del images
        full_proposals_ls = []
        cls_ls = []
        for cls_id, proposals in support_proposals_dict.items():
            full_proposals_ls.append(proposals[0])
            cls_ls.append(cls_id)
        
        # all prroposals boxes
        full_proposals_ls = [Instances.cat(full_proposals_ls)]
        proposal_boxes = [x.proposal_boxes for x in full_proposals_ls]
        # store all box features
        box_features = self._shared_roi_transform(
            [features["res4"]], proposal_boxes
        )
        
        full_scores_ls = []
        full_bboxes_ls = []
        full_cls_ls = []
        cnt = 0
        #for cls_id, support_box_features in support_box_features_dict.items():
        for cls_id in cls_ls:
            support_box_features = support_box_features_dict[cls_id]
            query_features = box_features[cnt*100:(cnt+1)*100]
            pred_class_logits, pred_proposal_deltas = self.box_predictor(query_features, support_box_features) # FsodFastRCNNOutputLayers
            # pred_class_logits, 特征图由头网络连接全连接层得到，预测的类别ID
            full_scores_ls.append(pred_class_logits)
            full_bboxes_ls.append(pred_proposal_deltas)
            # Returns a tensor with the same size as input filled with fill_value
            full_cls_ls.append(torch.full_like(pred_class_logits[:, 0].unsqueeze(-1), cls_id).to(torch.int8))
        
            del query_features
            del support_box_features

            cnt += 1
            
        # N*shape[0]
        class_logits = torch.cat(full_scores_ls, dim=0)  # (2000, 2) -> (20,2)
        
        proposal_deltas = torch.cat(full_bboxes_ls, dim=0)
        pred_cls = torch.cat(full_cls_ls, dim=0) #.unsqueeze(-1) 
       
        predictions = class_logits, proposal_deltas
        proposals = full_proposals_ls
        # # 每个图像中instance的得分    
        pred_instances, _ = self.box_predictor.inference(pred_cls, predictions, proposals)
        
        pred_instances = self.forward_with_given_boxes(features, pred_instances)
    
        return pred_instances, {}       
    
 #Returns-instances: (list[Instances]): A list of N instances, one for each image in the batch, 
 # that stores the topk most confidence detections.
    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        return instances
