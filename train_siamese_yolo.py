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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

EPOCHS = 20
BATCH = 3
MODEL_DEF = "config/yolov3-siamese.cfg"

def load_dataset():
    data_path = 'data/players/'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH,
        num_workers=0,
        shuffle=True
    )
    return train_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    model = Darknet(MODEL_DEF).to(device)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        for batch_i, (data, target) in enumerate(load_dataset()):
        # for batch_i in range(Batches):
            print("Batch {:d}".format(batch_i))
            # print("data:", data)
            # print("data size:", data.size())
            print("target:", target)
            # print("target size:", target.size())
            
            p1_count = torch.sum(target == 0)
            p1_class = data[target == 0, ..., ..., ...]
            p2_class = data[target != 0, ..., ..., ...]
            if p1_count == 3 or p1_count == 0:
                print("\tEmpty batch")
                continue
            elif p1_count > 1:
                anch = p1_class[0, ..., ..., ...].view(1, 3, 200, 200)
                pos = p1_class[1, ..., ..., ...].view(1, 3, 200, 200)
                neg = p2_class[0, ..., ..., ...].view(1, 3, 200, 200)
            else:
                anch = p2_class[0, ..., ..., ...].view(1, 3, 200, 200)
                pos = p2_class[1, ..., ..., ...].view(1, 3, 200, 200)
                neg = p1_class[0, ..., ..., ...].view(1, 3, 200, 200)

            # print("Anchor size:", anch.size())
            imgs = torch.cat([anch, pos, neg], 0)

            imgs = Variable(imgs.to(device))
            # print("Imgs size:", imgs.size())
        
            loss, outputs = model(imgs, targets=0)
            loss.backward()

            if batch_i % 9 == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

        #   Log progress
        print("Epoch {:d} --> loss {:.4f}".format(epoch, loss.item()))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "checkpoints/yolov3_id_ckpt_%d.pth" % epoch)
