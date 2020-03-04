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

def vectorDistance(v1, v2):
    if v1.size() == v2.size():
        #print("Comapring same vectors of size", v1.size())
        ret = torch.dist(v1, v2)
    else:
        #print("Comapring size", v1.size(), " and ", v2.size())
        ret = 1e6
    return ret


def vectorNN(vector, prev_vectors, selected, cls_pred, prev_cls):
    smallest = 1e3
    ret = -1
    for idx in range(len(prev_vectors)):
        #print("Comparing 'vector' with {:d}".format(idx))
        #print("Comparing 'vector'", vector, "with {:d}".format(idx), prev_vectors[idx])
        if not selected[idx] and int(cls_pred) == int(prev_cls[idx]):
            dist = vectorDistance(vector, prevImage_vectors[idx])
            if dist < smallest:
                smallest = dist
                ret = idx
    #print("-----------")
    #print("\t{:f}\t".format(smallest), end="")
    return ret

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.55, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--save_images", type=bool, default=True, help="flag if saving images with detections")
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
    vectors = []

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            #detections = model(input_imgs)
            (detections, all_vectors) = model(input_imgs, returnVectors=True)
            #detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            (detections, indices) = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, returnIndices=True)

          
        # Reshape vectors to correspond to indices
        for i in range(len(all_vectors)):
            #all_vectors[i] = all_vectors[i].reshape(opt.batch_size, int(1024/2**i), int(13*(2**i)*13*(2**i)))
            #print("vector {:d} resized...".format(i))
            all_vectors[i] = all_vectors[i].reshape(-1, int(1024/2**i), int(13*(2**i)*13*(2**i)))
            
        # Keep only valid vectors
        #print(indices)
        #vectors = [None for _ in range(len(indices))]
        batch_vectors = []
        for image_i in range(len(indices)):
            image_vectors = []
            for idx in indices[image_i]:
                real_idx = int( (int(idx)-1)/3 )
                if real_idx < 13**2:
                    anchor = 0
                elif real_idx < 26**2:
                    anchor = 1
                    real_idx -= 13**2
                else:
                    real_idx -= (13**2+26**2)
                    anchor = 2
                #print("Image {:d}, keeping vector {:d} from anchor {:d}".format(image_i, real_idx, 13*2**(anchor)))
                image_vectors.append(all_vectors[anchor][image_i, ..., real_idx])
            #print("Image vectors len:", len(image_vectors))
            batch_vectors.append(image_vectors)
            #print("-----")
        
        #print("Batch vectors len:", len(batch_vectors))
        vectors.append(batch_vectors)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    print("Vectors len: [{:d}, {:d}, {:d}]".format(len(vectors), len(vectors[0]), len(vectors[0][0])))
    
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    prevImage_vectors = []
    selected = []
    prev_cls = []
    
    IDs = list(range(25))
    oldIDs = list(range(25))
    nextID = 25
    num_dets = 0
    
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("({:02d}) Image: '{:s}'".format(img_i, path))

        prev_num_dets = num_dets
        num_dets = len(detections)
        add_cons = ((img_i)%2 * prev_num_dets)
        #print("-- Previous num_dets: {:d}, current num_dets {:d}".format(prev_num_dets, num_dets))
        #print("Add constant {:d}".format( add_cons ))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            img = np.array(Image.open(path))

            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            
            # Save detections into .txt file
            saveDetections(detections, path, img.shape, opt.conf_thres)
            
            # Do some magic with vectors
            image_vectors = vectors[int(img_i/opt.batch_size)][int(img_i%opt.batch_size)]
            if img_i > 0 and img_i%2 == 0:
                prevImage_vectors = vectors[int( (img_i-1)/opt.batch_size )][int( (img_i-1)%opt.batch_size )]
                prevImage_vectors += vectors[int( (img_i-2)/opt.batch_size )][int( (img_i-2)%opt.batch_size )]
                prev_cls = torch.cat((img_detections[img_i-2:img_i]), 0)[:, 6]
                #print("img_detections:", len(img_detections), img_detections[0])
                #print("Prev cls:", prev_cls)
                selected = [False for _ in range(len(prevImage_vectors))]
                #print("Image {:d}, number of vectors: {:d}, previous image {:d}".format(img_i, len(image_vectors), len(prevImage_vectors)))
                oldIDs = IDs.copy()
            
            
            print("IDs before change:", IDs)
            
            if len(prevImage_vectors) > 0:
                #print("Comparing image {:d} with another {:d} vectors".format(img_i, len(prevImage_vectors)))
                for idx in range(len(image_vectors)):
                    orig_idx = vectorNN(image_vectors[idx], prevImage_vectors, selected, detections[idx, 6], prev_cls)
                    #print("idx {:d}, changing cell {:d} ({:d}+{:d})".format(idx, idx + add_cons, idx, add_cons))
                    #print("-----")
                    if orig_idx < 0:
                        IDs[idx + add_cons] = nextID
                        nextID += 1
                    else:
                        selected[orig_idx] = True
                        IDs[idx + add_cons] = oldIDs[orig_idx]
                    #if int(detections[idx, 6]) == 4:
                        #print("idx:, {:d}, res {:d}, prev_cls:".format(idx, orig_idx), prev_cls)
                    #print("oldIDs:", oldIDs)
                    #print("IDs:", IDs)
                    #print("Res:", orig_idx)
                    #print("selected:", selected)
                    #print("-----")
                    #print("\t{:d}({:d}) --> {:d}({:d})".format(oldIDs[orig_idx], int(prev_cls[orig_idx]), idx+add_cons, int(detections[idx, 6])))
            #else:
                #print("Assigning new IDs to image {:d}".format(img_i))
                #correspondings = range(len(image_vectors))

            
            print("IDs to write:", IDs)
            if opt.save_images:
                
                # Create plot
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                unique_labels = detections[:, -2].cpu().unique()
                n_cls_preds = len(unique_labels)
                #bbox_colors = random.sample(colors, n_cls_preds)
                bbox_colors = [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0],
                                [1, 1, 0],
                                [0.5, 0.5, 0.5]]
                #for x1, y1, x2, y2, conf, cls_conf, cls_pred, ID in detections:
                for det_i in range(len(detections)):
                    x1, y1, x2, y2, conf, cls_conf, cls_pred, ID = detections[det_i]

                    if cls_conf.item() > 0.5:
                    
                        print("\t+ Label: %s, Conf: %.5f, ID: %d" % (classes[int(cls_pred)], cls_conf.item(), IDs[det_i + add_cons]))

                        box_w = max(x2 - x1, 20)
                        box_h = max(y2 - y1, 20)

                        #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        color = bbox_colors[int(cls_pred)]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        if int(ID) < 13**2 * 3:
                            anchor = 13
                        elif int(ID) < (13**2 + 26**2) * 3:
                            anchor = 26
                        else:
                            anchor = 52
                        plt.text(
                            x1,
                            y1,
                            s="{:d}".format(IDs[det_i + add_cons]),
                            #s="{:.2f}\n{:.2f}".format(conf, cls_conf),
                            #s=classes[int(cls_pred)],
                            color="black",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )


        if opt.save_images:
            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig("output/{}.png".format(filename), bbox_inches="tight", pad_inches=0.0)
        
            plt.close()
