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

import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import transforms
from scipy.spatial import distance_matrix, distance
from scipy.optimize import linear_sum_assignment

def saveDetections(dets, path, size, thresh, IDs, cons):
    #size = [1, 1]
    #print(size)
    name = path.split("/")[-1].split(".")[0]
    name = "output/"+name+".txt"
    f = open(name, 'w')
    for bb, bb_i in zip(dets, range(len(dets))):
        if bb[-2] > thresh:
            a = bb[0]
            b = bb[1]
            c = bb[2]
            d = bb[3]
            w = c-a
            h = d-b
            cx = a+w/2
            cy = b+h/2
            f.write("{:d} {:f} {:f} {:f} {:f} {:f}\n".format(IDs[bb_i + cons], cx/size[1], cy/size[0], w/size[1], h/size[0], bb[-2]))
    f.close()

def findNewOrOldDet(det, lost, sim, last_ID):
    # print("Trying to find new ID for detection number {:d}...".format(idx))
    tmp_ID = -1
    min_dist = 9999
    for key, value in lost.items():
        d = distance.pdist( [value, sim] )
        print("\tDistance {} for ID {:d}".format(d, key))
        if d < min_dist:
            min_dist = d
            tmp_ID = key
    if tmp_ID > -1:
        # print("Found old ID, assigning...")
        det.ID_type = "old"
        det.ID = tmp_ID
        del lost[key]
        print("Assigned old ID '{}' with sim {}".format(tmp_ID, min_dist))
    else:
        # print("No old ID found, making new one")
        det.ID = last_ID
        det.ID_type = "new"
        print("Assigned new ID '{}'".format(last_ID))
        last_ID += 1
    return det, lost, last_ID



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
    parser.add_argument("--save_detections", type=bool, default=True, help="flag if saving detections into txt files")
    parser.add_argument("--track", type=bool, default=True, help="flag if run tracking algorithm")
    parser.add_argument("--save_mot", type=bool, default=True, help="flag if saving detections into txt files")
    parser.add_argument("--save_images", type=bool, default=True, help="flag if saving images with detections")
    parser.add_argument("--save_field", type=bool, default=True, help="flag if saving images with detections")
    parser.add_argument("--save_cropped", type=bool, default=False, help="flag if saving cropped images of detected players")
    opt = parser.parse_args()
    
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    
    if opt.save_detections:
        os.makedirs(os.path.join("output", "detections"), exist_ok=True)
    
    if opt.save_images:
        os.makedirs(os.path.join("output", "images"), exist_ok=True)
    
    if opt.save_field:
        os.makedirs(os.path.join("output", "field"), exist_ok=True)

    if opt.save_cropped:
        os.makedirs(os.path.join("output", "cropped"), exist_ok=True)

    start_time = time.time()

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
            # print("vector {:d} resized...".format(i))
            all_vectors[i] = all_vectors[i].reshape(-1, int(256/2**i), int(52*(2**i)*52*(2**i)))
            # print("current size {}".format(all_vectors[i].size()))

        # Keep only valid vectors
        #vectors = [None for _ in range(len(indices))]
        batch_vectors = []
        for image_i in range(len(indices)):
            image_vectors = []
            # print(indices[image_i])
            for idx in indices[image_i]:
                real_idx = int( (int(idx)-1)/3 )
                if real_idx < 13**2:
                    anchor = 0
                elif real_idx < 26**2:
                    anchor = 1
                    real_idx -= 13**2
                else:
                    real_idx -= (13**2+26**2)
                    anchor = 0
                # print("Image {:d}, keeping vector {:d} from anchor {:d}".format(image_i, real_idx, 13*2**(anchor)))
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
    
    ## Delete model to free memory
    print("Deleting YOLO model to free memory...")
    del model

    ## Load Resnet, LDA and other stuff
    print("Importing Resnet and LDA...")
    import image_classes

    old_dets = None
    old_pos = None
    old_sim = None
    last_ID = 1
    lost = {}
    current_IDs = {}

    if opt.save_mot:
        mot_file = open("output/mot_output.txt", "w")
    
    print("\nSaving images Mira's way:")
    for img_i in range(0, len(imgs), 2):
        print("=====================================================================================")
        print("({:03d}) Images '{:s}' and '{:s}'".format(
            int(img_i/2),
            imgs[img_i].split("/")[-1],
            imgs[img_i+1].split("/")[-1]
        ))
        img_A = image_classes.image_with_detections(imgs[img_i], img_detections[img_i], False)
        img_B = image_classes.image_with_detections(imgs[img_i+1], img_detections[img_i+1], True)
        filenameA = img_A.path.split("/")[-1].split(".")[0]
        filenameB = img_B.path.split("/")[-1].split(".")[0]

        batch_number = int(img_i / 10)
        img_number = img_i % 10

        # Track
        if img_i == 0:
            # Init with random number
            first_id = random.randint(1, len(img_A.dets)+len(img_B.dets)-1)
            num = first_id
        else:
            # Track player by vector
            # print(vec)
            curr_vecs = []
            for v in vectors[batch_number][img_number]:
                curr_vecs.append(v.cpu().numpy().reshape(1, 256))
            for v in vectors[batch_number][img_number+1]:
                curr_vecs.append(v.cpu().numpy().reshape(1, 256))
            # print(vectors[batch_number][img_number])
            dst_mat = distance_matrix(vec, np.concatenate(curr_vecs))
            num = np.argmin(dst_mat)
            print("Distance matrix", dst_mat)
            print("Arg min", num)
        
        l = len(vectors[batch_number][img_number])
        if num >= l:
            print("num = {} - {} - 1".format(num, l))
            num -= l+1
            print("img_num = {} + 1".format(img_number))
            img_number += 1
            img_B.dets[num].cls_pred = 1
        else:
            img_A.dets[num].cls_pred = 1

        print("Selected player {} from image {}/{}".format(num, batch_number, img_number))
        vec = vectors[batch_number][img_number][num].cpu().numpy().reshape(1, 256)

        # Save detections
        if opt.save_detections:
            print("-----------------------------------------------")
            print("Saving detections...")
            img_A.save_detections(filenameA)
            img_B.save_detections(filenameB)

        # Save images
        if opt.save_images:
            print("-----------------------------------------------")
            img_A.draw_detections(filenameA, opt.save_cropped)
            print("   --------")
            img_B.draw_detections(filenameB, opt.save_cropped)

        # Save images of projection to field
        if opt.save_field:
            print("-----------------------------------------------")
            print("Saving real positions...")
            img_A.draw_positions(filenameA)
            img_B.draw_positions(filenameB)

        # Save tracking in MOT format
        if opt.save_mot:
            mot_file.write("{:s}".format(img_A.get_mot_format(img_i+1)))
            mot_file.write("{:s}".format(img_B.get_mot_format(img_i+2)))


    
    if opt.save_mot:
        mot_file.close()
    
    time_elapsed = time.time() - start_time
    inference_time = datetime.timedelta(seconds=time_elapsed)
    time_per_frame = datetime.timedelta(seconds=time_elapsed/len(imgs))
    print("Overall time: {}".format(inference_time))
    print("Number of frames: {:d}".format(len(imgs)))
    print("Time per frame: {}, fps: {:.1f}".format(time_per_frame, len(imgs)/time_elapsed))


        
    
    