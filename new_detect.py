from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from operator import itemgetter


def saveDetections(dets, path, size, thresh):
    #size = [1, 1]
    #print(size)
    name = path.split("/")[-1].split(".")[0]
    name = "output/"+name+".txt"
    f = open(name, 'w')
    for bb in dets:
        if bb[-2] > thresh:
            a = bb[0]
            b = bb[1]
            c = bb[2]
            d = bb[3]
            w = c-a
            h = d-b
            cx = a+w/2
            cy = b+h/2
            f.write("{:d} {:f} {:f} {:f} {:f} {:f}\n".format(int(bb[-1]), cx/size[1], cy/size[0], w/size[1], h/size[0], bb[-2]))
    f.close()

def saveDetectionsDual(dets, path1, path2, size, thresh):
    #size = [1, 1]
    #print(size)
    name1 = path1.split("/")[-1].split(".")[0]
    name1 = "output/dual/"+name1+".txt"
    name2 = path2.split("/")[-1].split(".")[0]
    name2 = "output/dual/"+name2+".txt"
    f1 = open(name1, 'w')
    f2 = open(name2, 'w')
    for bb in dets:
        a = bb[0]
        b = bb[1]
        c = bb[2]
        d = bb[3]
        w = c-a
        h = d-b
        cx = a+w/2
        cy = b+h/2
        if int(bb[7]) == 1:
            f1.write("{:d} {:f} {:f} {:f} {:f} {:f}\n".format(int(bb[6]), cx/size[1], cy/size[0], w/size[1], h/size[0], bb[5]))
        else:
            f2.write("{:d} {:f} {:f} {:f} {:f} {:f}\n".format(int(bb[6]), cx/size[1], cy/size[0], w/size[1], h/size[0], bb[5]))
    f1.close()
    f2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--save_images", type=bool, default=False, help="flag if saving images with detections")
    parser.add_argument("--dual_images", type=bool, default=True, help="flag if images are in KL/KP pairs")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    if opt.dual_images:
        for idx in range(0, len(imgs), 2):
            path1 = imgs[idx]
            path2 = imgs[idx+1]
            dets1 = img_detections[idx]
            dets2 = img_detections[idx+1]
            print("({:02d}) Images: '{:s}' and '{:s}'".format(idx, path1, path2))

            img1 = np.array(Image.open(path1))
            img2 = np.array(Image.open(path2))

            # Rescale boxes to original image
            dets1 = rescale_boxes(dets1, opt.img_size, img1.shape[:2])
            dets2 = rescale_boxes(dets2, opt.img_size, img2.shape[:2])

            detections = [[] for _ in range(4)]
            origin = torch.Tensor([1])
            for det in dets1:
                det = torch.cat((det, origin), -1)
                detections[int(det[-2])].append(det)
            origin = torch.Tensor([2])
            for det in dets2:
                det = torch.cat((det, origin), -1)
                detections[int(det[-2])].append(det)
                
            #for det in detections:
            #    print(det)

            limits = [5, 5, 2, 2]
            final_detections = []
            for k in range(4):
                count = 0
                while len(detections[k]) is not 0:
                    det = detections[k].pop(0)
                    final_detections.append(det)
                    count += 1
                    if count == limits[k]:
                        print("Discarding {:d} persons of class {:d}".format(len(detections[k]), k))
                        break
                if count < limits[k]:
                    print("Detected only {:d} persons of class {:d}".format(count, k))

            saveDetectionsDual(final_detections, path1, path2, img1.shape, opt.conf_thres)            
            
    else:
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("({:02d}) Image: '{:s}'".format(img_i, path))


            # Draw bounding boxes and labels of detections
            if detections is not None:
                img = np.array(Image.open(path))

                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            
                # Save detections into .txt file
                saveDetections(detections, path, img.shape, opt.conf_thres)
            
                if opt.save_images:
                
                    # Create plot
                    plt.figure()
                    fig, ax = plt.subplots(1)
                    ax.imshow(img)

                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    bbox_colors = [[1, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0],
                                    [1, 1, 0]]
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        if cls_conf.item() > 0.8:
                    
                            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                            box_w = max(x2 - x1, 20)
                            box_h = max(y2 - y1, 20)

                            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            # Create a Rectangle patch
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                            # Add the bbox to the plot
                            ax.add_patch(bbox)
                            # Add label
                            #plt.text(
                            #    x1,
                            #    y1,
                            #    s=classes[int(cls_pred)],
                            #    color="white",
                            #    verticalalignment="top",
                            #    bbox={"color": color, "pad": 0},
                            #)


            if opt.save_images:
                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                filename = path.split("/")[-1].split(".")[0]
                plt.savefig("output/{}.png".format(filename), bbox_inches="tight", pad_inches=0.0)
        
            plt.close()
