from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_CLASSES = 5

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        #print("Class", c, "--------")
        i = pred_cls == c
        #print("i:", i)
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        #print("n_gt:", n_gt)
        #print("n_p:", n_p)

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            #print("fpc:", fpc)
            #print("tpc:", tpc)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])
            #print("recall_curve:", recall_curve)

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])
            #print("precision_curve:", precision_curve)

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
            #print("ap:", ap[-1])

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 5]
        pred_labels = output[:, 6]

        #print("output:", output)
        #print("pred_boxes:", pred_boxes)
        #print("pred_scores:", pred_scores)
        #print("pred_labels:", pred_labels)

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        #print("annotations", annotations)
        #target_labels = annotations[:, 0] if len(annotations) else []
        target_labels = torch.Tensor((np.where(targets[:, 0:4] == 1)[1]).tolist()) if len(annotations) else []
        #print("target_labels:", target_labels)
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 5:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    #print("iou:", iou, "box_index:", box_index, "pred_i:", pred_i)
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    #print("Predictions size", prediction.size())
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    indices = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        v0 = np.arange(0, 13**2 * 3)
        v1 = np.arange(13**2 * 3, (13**2 + 26**2) * 3)
        v2 = np.arange((13**2 + 26**2) * 3, (13**2 + 26**2 + 52**2) * 3)
        v = np.concatenate((v0, v1, v2), axis=0)
        torch_v = torch.from_numpy(v)
        torch_v = torch_v[image_pred[:, 4] >= conf_thres]
        #print("image_pred indexes")
        #print(torch_v)
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        
        ## Change here!
        class_confs, class_preds = image_pred[:, 5:-1].max(1, keepdim=True)
        humaness = image_pred[:, -1].reshape(-1, 1)
        #print("class_confs.size():", class_confs.size())
        #print("class_preds.size():", class_preds.size())
        #print("humaness.size():", humaness.size())
        #print(image_pred)
        #print(class_confs, class_preds)
        #cls_confs = []
        #cls_preds = []
        #for det in image_pred:
        #    idx = torch.where(det[5:] > conf_thres)
        #    cls_preds += [idx]
        #    cls_confs += [det[5:][idx]]

        #print(cls_confs)
        #print(cls_preds)
        #print(image_pred[:, 5:])
        #print(image_pred[:, 5:][image_pred[:, 5:] > conf_thres])
        #print(image_pred[:, 5:] > conf_thres)
        #print(torch.where(image_pred[:, 5:] > conf_thres, 1))
        #print(image_pred[:, 5:])
        #print(image_pred[:, 5:] >= conf_thres)
        #print((image_pred[:, 5:][image_pred[:, 5:] > conf_thres]).reshape(-1, image_pred.size(0)))
        ##
        
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(), humaness.float()), 1)
        #print("Detections size", detections.size())
        # Perform non-maximum suppression
        keep_boxes = []
        keep_boxes_idx = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            huge_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > 0.6
            
            ## Here
            label_match = detections[0, -2] == detections[:, -2]
            ##

            '''
            if int(detections[0, -1]) < 1e-5:
                both_players = detections[:, -1] == 1
            elif int(detections[0, -1]) - 1 < 1e-5:
                both_players = detections[:, -1] == 0
            else:
                both_players = torch.BoolTensor([False for _ in range (detections.size(0))])
            '''
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            #invalid = large_overlap & (label_match | both_players)
            invalid = (large_overlap & label_match) | huge_overlap
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [torch.cat([detections[0], torch.Tensor([torch_v[0]])], 0)]
            keep_boxes_idx += [torch_v[0]]
            detections = detections[~invalid]
            #detections = detections[1:, :]
            torch_v = torch_v[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
            indices[image_i] = keep_boxes_idx
            #print("Keeping", output[image_i].size())
            #print(keep_boxes)

    return output
    #return output, indices


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    #print(pred_boxes)

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    #print("--------------------------------------------")
    #print("--------------------------------------------")
    #print("nB:", nB)
    #print("nA:", nA)
    #print("nC:", nC)
    #print("nG:", nG)
    #print("pred_boxes.size():", pred_boxes.size())
    #print("pred_cls.size():", pred_cls.size())

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, (1+nC):] * nG
    #target_boxes = target[:, (1+NUM_CLASSES):] * nG
    #print(target)
    #print(target_boxes)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    #print("\nbest_n:", best_n)
    # Separate target values
    tmp = target[:, :(1+nC)].long().t()
    b = tmp[0]
    target_labels = tmp[1:]
    #print("\ntmp:", tmp)
    #print("\nb:", b)
    #print("\ntarget_labels:", target_labels)
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    #print("\ngj:", gj)
    #print("\ngi:", gi)
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi] = torch.transpose(target_labels.float(), 0, 1)
    #print("\ntarget:", target)
    #print("\nb:", b)
    #print("\nbest_n:", best_n)
    #print("\ngj:", gj)
    #print("\ngi", gi)
    #print("\ntarget_labels", target_labels)
    #print("\ntcls size:", tcls.size())
    
    # Compute label correctness and iou at best anchor
    #test_target_labels = torch.cuda.BoolTensor(nB, nA, nG, nG, nC).fill_(False)
    #test_target_labels[b, best_n, gj, gi, target_labels] = True
    
    #print("\ntcls.size():", tcls.size())
    #print("\ntest:", test_target_labels[b, best_n, gj, gi])
    #print("\nmagic:", pred_cls[b, best_n, gj, gi])
    #print("\nmagic0:", pred_cls[b, best_n, gj, gi] > 0.5)
    #print("\nmagic1:", pred_cls[b, best_n, gj, gi].argmax(-1))
    #print("\nmagic2-new:", (pred_cls[b, best_n, gj, gi] > 0.5) == test_target_labels[b, best_n, gj, gi])
    #print("\nmagic2-new:", (torch.sum(((pred_cls[b, best_n, gj, gi] > 0.5) == test_target_labels[b, best_n, gj, gi]).float(), 1) == nC).float() )
    #print("\nmagic2:", pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels)
    #print("\nmagic3:", (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float())

    #print(tcls[b, best_n, gj, gi])

    #class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    class_mask[b, best_n, gj, gi] = (torch.sum(((pred_cls[b, best_n, gj, gi] > 0.5) == tcls[b, best_n, gj, gi].bool()).float(), 1) == nC).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
