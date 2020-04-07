import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import transforms
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

from utils.utils import rescale_boxes

classes = [
    "player_A",
    "player_B",
    "goalie",
    "ref",
    "human",
]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## Load Resnet18 and LDA for ID tracking
lda = pickle.load(open("LDA_model.sav", 'rb'))
resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
resnet.load_state_dict(torch.load("checkpoints/resnet18_id.pth"))
resnet.to('cuda')
resnet.eval()

class image_with_detections():
    def __init__(self, path, dets, flip):
        self.path = path
        self.img = np.array(Image.open(self.path))
        self.h, self.w = self.img.shape[:2]
        self.flip = flip
        self.add_detections(dets)
        self.thresh = 0.5   # Threshold when accept detection for saving and draving 
        self.detections_output_folder = os.path.join("output", "detections")
        self.images_output_folder = os.path.join("output", "images")
        self.cropped_output_folder = os.path.join("output", "cropped")
        self.colors = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0.5, 0.5, 0.5]
        ]
    
    def add_detections(self, dets):
        dets = rescale_boxes(dets, 416, self.img.shape[:2])
        self.dets = []
        for d in dets:
            det = detection_with_info(self.img, d)
            det.set_position(self.flip)
            self.dets.append(det)   

    def save_detections(self, filename):
        name = "{}.txt".format(filename)
        f = open(os.path.join(self.detections_output_folder, name), 'w')
        for det in self.dets:
            det_str= det.get_detection_string(self.thresh)
            if det_str is not None:
                f.write("{:s}\n".format(det_str))
        f.close()

    def draw_detections(self, filename, save_cropped=False):
        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(self.img)

        for det in self.dets:
            if det.cls_conf > self.thresh:
                # print("\t+ Label: %s, Conf: %.5f" % (classes[det.cls_pred], det.cls_conf))
                if det.ID_type:
                    print("\t+ Label: {:10s}\t conf: {:.4f}\t ID:{:3d}\t ID_type: {:s}".format(classes[det.cls_pred], det.cls_conf, det.ID, det.ID_type))
                else:
                    print("\t+ Label: {:10s}\t conf: {:.4f}\t ID:{:3d}" .format(classes[det.cls_pred], det.cls_conf, det.ID))

                box_w = max(det.w, 20)
                box_h = max(det.h, 20)

                #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color = self.colors[det.cls_pred]
                # Create a Rectangle patch
                bbox = patches.Rectangle((det.x1, det.y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    det.x1,
                    det.y1,
                    # s="",
                    s="{:d}_{:s}".format(det.ID, det.ID_type) if det.ID_type else "{:d}".format(det.ID),
                    color="black",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
            if save_cropped:
                name = "p_{:03d}_img_{:s}.png".format(det.ID, filename)
                path = os.path.join(self.cropped_output_folder, name)
                det.cropped.save(path)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        path = os.path.join(self.images_output_folder, "{}.png".format(filename))
        plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')

    def get_positions_list(self):
        pos = []
        for d in self.dets:
            pos.append(d.position)
        return pos
    
    def get_ID_list(self):
        pos = []
        for d in self.dets:
            pos.append(d.ID)
        return pos
    
    def get_LDA_list(self):
        pos = []
        for d in self.dets:
            pos.append(d.get_LDA())
        return pos


class detection_with_info():
    def __init__(self, img, det):
        x1, y1, x2, y2, conf, cls_conf, cls_pred, ID = det
        # Info from YOLO
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.cls_pred = int(cls_pred)
        self.cls_conf = float(cls_conf)
        self.conf = float(conf)

        # Image info
        # Crop image
        self.img_h, self.img_w = img.shape[:2]
        area = (max(self.x1, 0), max(self.y1, 0), min(self.x2, self.img_w), min(self.y2, self.img_h))
        # print(self.x1, self.y1, self.x2, self.y2)
        # print(area)
        self.cropped = Image.fromarray(img, 'RGB').crop(area)
        # img_name = "players/p_{:03d}_img_{:03d}.png".format(IDs[det_i + add_cons], img_i)
        # cropped.save(img_name)

        # Crop rescale, normalize
        self.img = preprocess(self.cropped).type(torch.cuda.FloatTensor)

        # Computed info
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.position = None
        self.ID = -1
        self.ID_type = None

    def set_position(self, flipped):
        self.position = [self.x1 + self.w, self.y1 + self.h]
        if flipped:
            self.position[0] = self.img_w - self.position[0]
            self.position[1] = 2*self.img_h - self.position[1]
        else:
            self.position = [self.x1 + self.w, self.y1 + self.h]

    def get_LDA(self):
        embed = resnet(self.img.to('cuda').view(1, 3, 224, 224))
        lda_tmp = lda.transform(embed.detach().to('cpu'))
        return [lda_tmp[0][0], lda_tmp[0][1]]

    def get_detection_string(self, thresh=0.5):
        s = None
        if self.cls_conf > thresh:
            s = "{:d} {:f} {:f} {:f} {:f} {:d}".format(
                self.ID,                            # ID
                (self.x1 + self.h/2)/self.img_h,    # X coordimate of center, normalized
                (self.y1 + self.w/2)/self.img_w,    # Y coordinate of center, normalized
                self.w/self.img_w,                  # BB width, normalized
                self.h/self.img_h,                  # BB height, normalized
                self.cls_pred                       # Predicted class
            )
        return s
        