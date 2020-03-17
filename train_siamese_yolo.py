from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import random
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

EPOCHS = 2000
BATCH = 5
MODEL_DEF = "config/yolov3-siamese.cfg"
SIZE = 200
LR = 1e-4

def load_dataset():
    data_path = 'data/players/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose( [transforms.Resize(SIZE), transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def testModel(model, p1_img, p2_img):
    ''' Prints table of distances
    inputs:
        p1, p2: encoding for all images in format [nImages, vectorLen]
    '''
    p1 = model(p1_img)
    p2 = model(p2_img)

    p1_num = p1.size(0)
    p2_num = p2.size(0)
    vectorLen = p1.size(1)

    print("p1\\p2\t", end="")
    for p2_i in range(p2_num):
        print("{:02d}\t".format(p2_i), end="")
    print("\n--------------------------------------------------------------------------------------------------------------------------")


    for p1_i in range(p1_num):
        print("{:02d}|\t".format(p1_i), end="")
        for p2_i in range(p2_num):
            print("{:6.2f}\t".format(torch.dist(p1[p1_i, ...].view(1, vectorLen), p2[p2_i, ...].view(1, vectorLen))), end="")
        print()
    print("--------------------------------------------------------------------------------------------------------------------------")
    for p1_i in range(p1_num):
        print("{:02d}|\t".format(p1_i), end="")
        for p1_j in range(p1_i):
            print("{:6.2f}\t".format(torch.dist(p1[p1_i, ...].view(1, vectorLen), p1[p1_j, ...].view(1, vectorLen))), end="")
        print()
    print("--------------------------------------------------------------------------------------------------------------------------")
    for p2_i in range(p2_num):
        print("{:02d}|\t".format(p2_i), end="")
        for p2_j in range(p2_i):
            print("{:6.2f}\t".format(torch.dist(p2[p2_i, ...].view(1, vectorLen), p2[p2_j, ...].view(1, vectorLen))), end="")
        print()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    yolo = Darknet(MODEL_DEF).to(device)
    yolo.apply(weights_init_normal)

    resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
    resnet.to('cuda')
    loss_fcn = torch.nn.TripletMarginLoss(margin=200, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='none')

    # model = yolo
    model = yolo

    print("===== Model architecture =====")
    print(model)
    print("==============================")

    p1_filelist = glob.glob("data/players/p1/*.jpg")
    p2_filelist = glob.glob("data/players/p2/*.jpg")
    player1 = torch.from_numpy(np.array([np.array(Image.open(fname)) for fname in p1_filelist])).type(torch.cuda.FloatTensor)
    player2 = torch.from_numpy(np.array([np.array(Image.open(fname)) for fname in p2_filelist])).type(torch.cuda.FloatTensor)

    player1 = player1.permute(0, 3, 1 ,2)
    player2 = player2.permute(0, 3, 1 ,2)

    print("Player 1:", player1.size())
    print("Player 2:", player2.size())

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(resnet.parameters())

    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        # for batch_i, (data, target) in enumerate(load_dataset()):
        for batch_i in range(BATCH):
            # print("Batch {:d}".format(batch_i))
            # print("data:", data)
            # print("data size:", data.size())
            # print("target size:", target.size())
            
            idx = random.randint(0, 1)
            a = random.randint(0, 4)
            p = random.randint(0, 4)
            while a==p:
                p = random.randint(0, 4)
            n = random.randint(0, 4)

            idx = 0
            a = 0
            p = 2
            n = 3

            if idx < 1:
                anch = player1[a, ..., ..., ...].view(1, 3, SIZE, SIZE)
                pos = player1[p, ..., ..., ...].view(1, 3, SIZE, SIZE)
                neg = player2[n, ..., ..., ...].view(1, 3, SIZE, SIZE)
            else:
                anch = player2[a, ..., ..., ...].view(1, 3, SIZE, SIZE)
                pos = player2[p, ..., ..., ...].view(1, 3, SIZE, SIZE)
                neg = player1[n, ..., ..., ...].view(1, 3, SIZE, SIZE)
                
            # p1_count = torch.sum(target == 0)
            # p1_class = data[target == 0, ..., ..., ...]
            # p2_class = data[target != 0, ..., ..., ...]
            # # print("target:", target, end="")
            # if p1_count == 3 or p1_count == 0:
            #     # print("\tEmpty batch")
            #     continue
            # elif p1_count > 1:
            #     anch = p1_class[0, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     pos = p1_class[1, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     neg = p2_class[0, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     # print("Anchor is class 0")
            # else:
            #     anch = p2_class[0, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     pos = p2_class[1, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     neg = p1_class[0, ..., ..., ...].view(1, 3, SIZE, SIZE)
            #     # print("Anchor is class 1")

            # print("Anchor size:", anch.size())
            imgs = torch.cat([anch, pos, neg], 0)

            imgs = Variable(imgs.to(device))
            # print("Imgs size:", imgs.size())
        
            if model == yolo:
                loss, outputs = model(imgs, targets=0)
            else:
                outputs = model(imgs)
                anchor = outputs[0, ...].view(1, outputs.size(1))
                positive = outputs[1, ...].view(1, outputs.size(1))
                negative = outputs[2, ...].view(1, outputs.size(1))
                loss = loss_fcn(anchor, positive, negative)
            
            loss.backward()

            
            #   Log progress
            # print("---")
            # print("|\tloss {:.4f}".format(loss.item()))

            # dist_ap = torch.dist(anchor, positive).item()
            # dist_an = torch.dist(anchor, negative).item()
            # dist_pn = torch.dist(positive, negative).item()
            # print("|\tidx: {:d}, a: {:d}, p: {:d}, n: {:d}".format(idx, a, p, n))
            # print("|\tD(a,p): {:.1f}, D(a,n): {:.1f}, D(p,n): {:.4f}".format(dist_ap, dist_an, dist_pn))
            # print("|\tD(a,p) - D(a,n) = {:.1f}".format(dist_ap-dist_an))
            # print("---")

            if (batch_i+1) % BATCH == 0:
                # with torch.no_grad():
                #     for param in model.parameters():
                #         param -= LR * param.grad
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

        # print("Epoch {:d}".format(epoch))

        if epoch % 20 == 0:
            print("================================================= vv epoch {:d} vv ===============================================================".format(epoch))
            testModel(model, player1[..., ..., ..., ...], player2[..., ..., ..., ...])
            print("================================================= ^^ epoch {:d} ^^ ===============================================================".format(epoch))
        #     torch.save(model.state_dict(), "checkpoints/yolov3_id_ckpt_%d.pth" % epoch)

    # p1_encodings = resnet(player1)
    # p2_encodings = resnet(player2)
    # testModel(p1_encodings, p2_encodings)
