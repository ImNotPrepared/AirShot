# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from .locator import LocatorNet

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

__all__ = ["fsod_fast_rcnn_inference", "FsodFastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fsod_fast_rcnn_inference(pred_cls, boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fsod_fast_rcnn_inference_single_image` for all images.

    Args:
        !!!!!!! Ri is the number of predicted objects for image i !!!!!!!
        !!!!!!! K is the categories !!!!!!!
        
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FsodFastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FsodFastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fsod_fast_rcnn_inference_single_image(
            pred_cls_per_image, boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for pred_cls_per_image, scores_per_image, boxes_per_image, image_shape in zip(pred_cls, scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fsod_fast_rcnn_inference_single_image(
    pred_cls, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fsod_fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fsod_fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    indices = torch.arange(start=0, end=scores.shape[0], dtype=int)
    indices = indices.expand((scores.shape[1], scores.shape[0])).T

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        pred_cls = pred_cls[valid_mask]
        indices = indices[valid_mask]

    scores = scores[:, :-1]
    indices = indices[:, :-1]
    
    cls_num = pred_cls.unique().shape[0]
    box_num = int(scores.shape[0] / cls_num)

    scores = scores.reshape(cls_num, box_num).permute(1, 0)
    indices = indices.reshape(cls_num, box_num).permute(1, 0)
    boxes = boxes.reshape(cls_num, box_num, 4).permute(1, 0, 2).reshape(box_num, -1)
    pred_cls = pred_cls.reshape(cls_num, box_num).permute(1, 0)

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero() #  return indices where is not 0, shape RxK
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    indices = indices[filter_mask]
    pred_cls = pred_cls[filter_mask]

    # Apply per-class NMS
    '''    Performs non-maximum suppression in a batched fashion.
    
    Decreasing order of scores

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold'''
        
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image] # keep top k score
    #boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    boxes, scores, filter_inds, pred_cls = boxes[keep], scores[keep], filter_inds[keep], pred_cls[keep]
    indices = indices[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    #result.pred_classes = filter_inds[:, 1]
    result.pred_classes = pred_cls
    result.indices = indices

    return result, filter_inds[:, 0] # shape R*K -> R *  1


class FsodFastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.image_shapes = [x.image_size for x in proposals]
        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        
        num_instances = self.gt_classes.numel() 
        

        pred_classes = self.pred_class_logits.argmax(dim=1) 
        bg_class_ind = self.pred_class_logits.shape[1] - 1 # [256, 2]
        
        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        
        num_fg = fg_inds.nonzero().numel()
        
        fg_gt_classes = self.gt_classes[fg_inds] # [0,0,....0]
        
        fg_pred_classes = pred_classes[fg_inds]
        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()


        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()

        num_instances = self.gt_classes.numel()
        # gt cls
        cls_score_softmax = F.softmax(self.pred_class_logits, dim=1)  # [256,2]
        fg_inds = (self.gt_classes == 0).nonzero().squeeze(-1)
        bg_inds = (self.gt_classes == 1).nonzero().squeeze(-1)
        # all 1 
        # fg: [0-32], bg: [32-255]
        bg_cls_score_softmax = cls_score_softmax[bg_inds, :]

        bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(num_instances * 0.25))) #int(num_instances * 0.5 - fg_inds.shape[0])))
        bg_num_1 = max(1, min(fg_inds.shape[0] * 1, bg_num_0))
        # 2:1
        sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 0], descending=True)
        real_bg_inds = bg_inds[sorted_bg_inds]
        real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(num_instances * 0.5)][:bg_num_0]
        real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(num_instances * 0.5)][:bg_num_1]

        topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
        
        #print(self.pred_class_logits.shape) -> (256,2)   print(self.gt_classes.shape) -> (256)
        
    
        CLASS_Loss = F.cross_entropy(self.pred_class_logits[topk_inds], self.gt_classes[topk_inds])
        POS_Loss = F.cross_entropy(self.pred_class_logits[fg_inds], self.gt_classes[fg_inds]) 
        NEG_Loss =  F.cross_entropy(self.pred_class_logits[real_bg_topk_inds_1], self.gt_classes[real_bg_topk_inds_1]) 
        BG_Loss =  F.cross_entropy(self.pred_class_logits[real_bg_topk_inds_0], self.gt_classes[real_bg_topk_inds_0]) 
        
        return [CLASS_Loss, POS_Loss, NEG_Loss, BG_Loss]#, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()
        
        
        # def get_deltas(self, src_boxes, target_boxes)
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim #  if dim1=dim2 get true
        device = self.pred_proposal_deltas.device


        #  pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
        #        logits for all R predicted object instances.
        #        Each row corresponds to a predicted object instance.
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        # foreground indices
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
            
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel() #  numel get the num
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
    ### sequence of apply_deltas  def apply_deltas(self, deltas, boxes):

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "POS_Loss":self.softmax_cross_entropy_loss()[1],
            "NEG_Loss": self.softmax_cross_entropy_loss()[2],
            "BG_Loss": self.softmax_cross_entropy_loss()[3],
            "loss_cls": self.softmax_cross_entropy_loss()[0],
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fsod_fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class FsodFastRCNNOutputLayers(nn.Module):
    """
    
    used  in:
             
    self.box_predictor = FsodFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape) ## used for specifying the shape.
        
        # size = channel * width * height
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)

        box_dim = len(box2box_transform.weights)
        # few shot
        self.patch_relation = True
        self.local_correlation = True
        self.global_relation = True
        self.locator = LocatorNet(7, input_shape.channels, 1024)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        dim_in = input_size
        
        
        if self.patch_relation:
            self.conv_1 = nn.Conv2d(dim_in*2, int(dim_in/4), 1, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(int(dim_in/4), int(dim_in/4), 3, padding=0, bias=False)
            self.conv_3 = nn.Conv2d(int(dim_in/4), dim_in, 1, padding=0, bias=False)
            # self.bbox_pred_pr = nn.Linear(dim_in, 4) #num_bbox_reg_classes * box_dim)
            self.cls_score_pr = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)
 
        if self.local_correlation:
            self.conv_cor = nn.Conv2d(dim_in, dim_in, 1, padding=0, bias=False)
            #self.bbox_pred_cor = nn.Linear(dim_in, 4)
            self.cls_score_cor = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)

        if self.global_relation:
            self.fc_1 = nn.Linear(dim_in * 2, dim_in)
            self.fc_2 = nn.Linear(dim_in, dim_in)
            #self.bbox_pred_fc = nn.Linear(dim_in, 4)
            self.cls_score_fc = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)

        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
        self.avgpool_fc = nn.AvgPool2d(7)

        if self.patch_relation:
            nn.init.normal_(self.conv_1.weight, std=0.01)
            nn.init.normal_(self.conv_2.weight, std=0.01)
            nn.init.normal_(self.conv_3.weight, std=0.01)
            nn.init.normal_(self.cls_score_pr.weight, std=0.01)
            nn.init.constant_(self.cls_score_pr.bias, 0)
            # nn.init.normal_(self.bbox_pred_pr.weight, std=0.001)
            # nn.init.constant_(self.bbox_pred_pr.bias, 0)

        if self.local_correlation:
            nn.init.normal_(self.conv_cor.weight, std=0.01)
            nn.init.normal_(self.cls_score_cor.weight, std=0.01)
            nn.init.constant_(self.cls_score_cor.bias, 0)
            #init.normal_(self.bbox_pred_cor.weight, std=0.001)
            #init.constant_(self.bbox_pred_cor.bias, 0)

        if self.global_relation:
            nn.init.normal_(self.fc_1.weight, std=0.01)
            nn.init.constant_(self.fc_1.bias, 0)
            nn.init.normal_(self.fc_2.weight, std=0.01)
            nn.init.constant_(self.fc_2.bias, 0)
            nn.init.normal_(self.cls_score_fc.weight, std=0.01)
            nn.init.constant_(self.cls_score_fc.bias, 0)
        ###########################################################

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE
            # fmt: on
        }

    def forward(self, x_query, x_support):
        support = x_support #.mean(0, True) # avg pool on res4 or avg pool here?
        # classifier
        # fc
        if self.global_relation:
            x_query_fc = self.avgpool_fc(x_query).squeeze(3).squeeze(2)
            support_fc = self.avgpool_fc(support).squeeze(3).squeeze(2).expand_as(x_query_fc)
            cat_fc = torch.cat((x_query_fc, support_fc), 1)
            out_fc = F.relu(self.fc_1(cat_fc), inplace=True)
            out_fc = F.relu(self.fc_2(out_fc), inplace=True)
            cls_score_fc = self.cls_score_fc(out_fc)

        # correlation
        if self.local_correlation:
            x_query_cor = self.conv_cor(x_query)
            support_cor = self.conv_cor(support)
            x_cor = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=2048), inplace=True).squeeze(3).squeeze(2)
            cls_score_cor = self.cls_score_cor(x_cor)

        # relation
        if self.patch_relation:
            support_relation = support.expand_as(x_query)
            x = torch.cat((x_query, support_relation), 1)
            x = F.relu(self.conv_1(x), inplace=True) # 5x5
            x = self.avgpool(x)
            x = F.relu(self.conv_2(x), inplace=True) # 3x3
            x = F.relu(self.conv_3(x), inplace=True) # 3x3
            x = self.avgpool(x) # 1x1
            x = x.squeeze(3).squeeze(2)
            cls_score_pr = self.cls_score_pr(x)

        # box regressior
        bbox_pred_all = self.locator(x_query, x_support)
        # final result
        #storage = get_event_storage()
        
        cls_score_all = cls_score_pr + cls_score_cor + cls_score_fc # [128, 2]
        return cls_score_all, bbox_pred_all

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :  meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        return FsodFastRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).losses()

    def inference(self, pred_cls, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fsod_fast_rcnn_inference`.
            list[Tensor]: same as `fsod_fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals) # get box 
        scores = self.predict_probs(predictions, proposals) # get probability 

        num_inst_per_image = [len(p) for p in proposals]
        pred_cls = pred_cls.split(num_inst_per_image, dim=0) # 每个图像中instance的得分
        image_shapes = [x.image_size for x in proposals]
        return fsod_fast_rcnn_inference(
            pred_cls,
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _ = predictions  # get all the scores 
        num_inst_per_image = [len(p) for p in proposals]  #  numer  of instance 
        probs = F.softmax(scores, dim=-1) # softmax for each row 
        return probs.split(num_inst_per_image, dim=0) # list of sizes for each chunk
