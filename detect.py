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

        # Track
        if opt.track:
            pos = img_A.get_positions_list() + img_B.get_positions_list()
            sim = img_A.get_LDA_list() + img_B.get_LDA_list()
            dets = img_A.dets + img_B.dets

            # Assign IDs
            if old_pos:
                
                dist_mat = distance_matrix(pos, old_pos)
                # print(sim)
                sim_mat = distance_matrix(sim, old_sim)
                dist_mat[dist_mat>100] = 1000   
                sim_mat[sim_mat>200] = 900   
                com_mat = sim_mat + dist_mat
                # print("dist_mat:")
                # print(dist_mat)
                # print("sim_mat:")
                # print(sim_mat)
                # com_mat = sim_mat
                control_com, com_asg = linear_sum_assignment(com_mat)
                
                # print("control_com:", control_com)
                # print("com_asg:", com_asg)

                for idx in range(len(dets)):
                    if idx in control_com:
                        cost_a = com_asg[ int(np.where(control_com == idx)[0]) ]
                        cost = com_mat[idx][cost_a]
                        
                        if cost > 1000: 
                            dets[idx], lost, last_ID = findNewOrOldDet(dets[idx], lost, sim[idx], last_ID)
                            current_IDs[dets[idx].ID] = idx
                            # del prev_IDs[dets[idx].ID]
                        else:
                            dets[idx].ID = old_dets[cost_a].ID
                            current_IDs[dets[idx].ID] = idx
                            # del prev_IDs[dets[idx].ID]
                            # print("Old idx {} --> idx {}".format(cost_a, idx))
                        # print("Detection {:02d} assigned (cost_a: {:02d}, ID: {:02d}) by cost ->\t{:.2f} ({:.2f}, {:.2f})".format(
                            # idx,
                            # cost_a,
                            # dets[idx].ID,
                            # cost,
                            # dist_mat[idx][cost_a],
                            # sim_mat[idx][cost_a],
                            # ))
                    else:
                        dets[idx], lost, last_ID = findNewOrOldDet(dets[idx], lost, sim[idx], last_ID)
                        current_IDs[dets[idx].ID] = idx
                        # del prev_IDs[dets[idx].ID]

                # Delete all current IDs
                for key, value in current_IDs.items():
                    if key in prev_IDs:
                        del prev_IDs[key
                        ]
                
                for key, value in prev_IDs.items():
                    print("Loosing ID '{}' on position {}".format(key, value))
                    lost[key] = old_sim[value]
                
                print("Lost IDs:")
                print(prev_IDs)
                # print(current_IDs)



            else:
                for i, d in enumerate(dets):
                    d.ID = last_ID
                    current_IDs[last_ID] = i
                    d.ID_type = "new"
                    last_ID += 1
       
            # Transfer new dets to old etc.
            old_dets = dets
            old_pos = pos
            old_sim = sim
            prev_IDs = current_IDs.copy()
            current_IDs = {}
            dets = None
        
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


        
    
    