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
from PIL import Image, ImageOps

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


EPOCHS = 200
BATCH = 1
NUM_TRAIN_PLAYERS = 7
NUM_TEST_PLAYERS = 10
TRAIN = 0.7        # Percentage of training set (including validation)
USED = 1       # Percentage of unused images (to avoid CUDA-out-of-memory error)
MODEL_DEF = "config/yolov3-siamese.cfg"
SIZE = 224
LR = 1e-4

TRAIN_PLAYERS = random.sample(range(10), NUM_TRAIN_PLAYERS)
TEST_PLAYERS = random.sample(range(10), NUM_TEST_PLAYERS)

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
        print("{:02d}\t".format(p2_i+1), end="")
    print("\n--------------------------------------------------------------------------------------------------------------------------")


    for p1_i in range(p1_num):
        print("{:02d}|\t".format(p1_i+1), end="")
        for p2_i in range(p2_num):
            print("{:6.2f}\t".format(torch.dist(p1[p1_i, ...].view(1, vectorLen), p2[p2_i, ...].view(1, vectorLen)).item()), end="")
        print()
    print("--------------------------------------------------------------------------------------------------------------------------")
    for p1_i in range(p1_num):
        print("{:02d}|\t".format(p1_i+1), end="")
        for p1_j in range(p1_i):
            print("{:6.2f}\t".format(torch.dist(p1[p1_i, ...].view(1, vectorLen), p1[p1_j, ...].view(1, vectorLen)).item()), end="")
        print()
    print("--------------------------------------------------------------------------------------------------------------------------")
    for p2_i in range(p2_num):
        print("{:02d}|\t".format(p2_i+1), end="")
        for p2_j in range(p2_i):
            print("{:6.2f}\t".format(torch.dist(p2[p2_i, ...].view(1, vectorLen), p2[p2_j, ...].view(1, vectorLen)).item()), end="")
        print()

def computeClassDist(p1, p2):
    p1_num = p1.size(0)
    p2_num = p2.size(0)
    vectorLen = p1.size(1)
    d = []
    for p1_i in range(p1_num):
        for p2_i in range(p2_num):
            d.append(torch.dist(p1[p1_i, ...].view(1, vectorLen), p2[p2_i, ...].view(1, vectorLen)))
    d = torch.stack(d, 0)
    
    mx = float(torch.max(d))
    avg = float(torch.mean(d))
    md = float(torch.median(d))
    return [md, avg, mx]

def testmodelShort(model, players):
    p_enc = []
    p_size = []
    inter = []
    for p in players:
        p_tmp = model(p)
        p_enc.append(p_tmp)
        p_size.append(p.size(0))
        # Compute distance in classes

    # Compute distances between classes
    print("=======================================") 
    for p_i in range(len(p_enc)):
        md, avg, mx = computeClassDist(p_enc[p_i], p_enc[p_i])
        print("Inter class ({:d}):\n\t[{:0.2f}, {:0.2f}, {:0.2f}]".format(p_i+1, md, avg, mx))
        for p_j in range(p_i+1, len(p_enc)):
            md, avg, mx = computeClassDist(p_enc[p_i], p_enc[p_j])
            print("{:d} vs {:d}:\n\t[{:0.2f}, {:0.2f}, {:0.2f}]".format(p_i+1, p_j+1, md, avg, mx))
        print("=======================================") 

def visualizePCA(model, players, fname):
    markers = [".", "+", "x", "*", "1", "s", "v"]
    leg = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8"]
    pca = PCA(n_components=2)
    plt.clf()
    for i, p in enumerate(players):
        reduced = pca.fit_transform(model(p).cpu().detach())       
        t = reduced.transpose()
        plt.scatter(t[0], t[1], marker=markers[i])
    plt.legend(leg[:len(players)])
    plt.savefig(fname, format="jpg")
    print("PCA '{}' done!".format(fname))

def visualizeLDA(model, players, fname):
    markers = [".", "+", "x", "*", "s", "v", "^", "p", "D", "1", "2", "3", "4"]
    markers = [".", ".", ".", ".", ".", "+", "+", "+", "+", "+", "+"]
    markers_train = ["o", "o", "o", "o", "o", "P", "P", "P", "P", "P", "P"]
    leg = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8", "Player 9", "Player 10"]
    
    vectors = []
    num = [p.size(0) for p in players]
    tmp = []
    for p in players:
        i = 0
        while i < p.size(0):
            e = min(i+50, p.size(0))
            t = p[i:e, ...]
            tmp.append(model(t).cpu().detach())
            i = e

    X = torch.cat(tmp).numpy()
    # X = torch.cat([model(p).cpu().detach() for p in players]).numpy()
    
    y = []
    for i, n in enumerate(num):
        y += [i] * n
    y = np.array(y)
    # print("gt:", y)

    lda = LinearDiscriminantAnalysis(n_components=2)
    new_X = lda.fit_transform(X, y)
    plt.clf()
    if len(players) > 2:
        s = 0
        for i, n in enumerate(num):
            if i in TRAIN_PLAYERS:
                plt.scatter(new_X[s:(s+n), 0], new_X[s:(s+n), 1], marker=markers_train[i])
            else:
                plt.scatter(new_X[s:(s+n), 0], new_X[s:(s+n), 1], marker=markers[i])
            s += n
    else:
        s = 0
        for i, n in enumerate(num):
            if i in TRAIN_PLAYERS:
                plt.scatter(new_X[s:(s+n)], new_X[s:(s+n)], marker=markers_train[i])
            else:
                plt.scatter(new_X[s:(s+n)], new_X[s:(s+n)], marker=markers[i])
            s += n
        
    plt.legend(leg[:len(players)])
    plt.savefig(fname, format="jpg")

    
    print("LDA '{}' done!".format(fname))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("LDA", exist_ok=True)
    for path in glob.glob("PCA_*.jpg"):
        os.remove(path)
    for path in glob.glob("LDA/LDA_epoch_*.jpg"):
        os.remove(path)

    # Initiate model
    yolo = Darknet(MODEL_DEF).to(device)
    yolo.apply(weights_init_normal)

    resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
    resnet.to(device)
    loss_fcn = torch.nn.TripletMarginLoss(margin=100, p=2.0, eps=1e-02, swap=False, size_average=None, reduce=None, reduction='none')

    model = resnet
    # model = yolo

    print("===== Model architecture =====")
    print(model)
    print("==============================")

    players = []
    test_players = []
    indices = []
    
    preprocess = preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for j in range(10):
        i = j+5 if j > 4 else j
        file_list = glob.glob("data/players/p{:02d}/*.png".format(i+1))
        player = torch.stack([preprocess(Image.open(fname)) for fname in file_list]).type(torch.cuda.FloatTensor)
        num_used = int(player.size(0) * USED)
        player = player[:num_used, ..., ..., ...]
        num_images = player.size(0)
        num_train_set = int(TRAIN * num_images)
        idx = torch.zeros(num_images, dtype=torch.bool)
        idx[random.sample(range(num_images), num_train_set)] = True
        train = player[idx, ..., ..., ...]
        test = player[~idx, ..., ..., ...]
        print("Player {:d}".format(i+1))
        if j in TRAIN_PLAYERS:
            print("\ttrain size: {}".format(train.size()))
            players.append(train)
            indices.append(list(range(train.size(0))))
        if j in TEST_PLAYERS:
            print("\ttest size: {}".format(test.size()))
            test_players.append(test)

    optimizer = torch.optim.Adam(model.parameters())

    start_time = time.time()
    sum_loss = 0
    learning_set = [0, 9]
    max_len = 100
    a_indices = list(range(learning_set[0], learning_set[1]+1))
    p_indices = list(range(learning_set[0], learning_set[1]+1))
    n_indices = list(range(learning_set[0], learning_set[1]+1))

    visualizeLDA(model, test_players, "LDA/LDA_prior.jpg")

    for epoch in range(EPOCHS):
        epoch_time = time.time()
        model.train()
        # for batch_i, (data, target) in enumerate(load_dataset()):
        for batch_i in range(BATCH):
            # print("Batch {:d}".format(batch_i))
            # print("data:", data)
            # print("data size:", data.size())
            # print("target size:", target.size())
            
            idx, idx_neg = random.sample(range(len(players)), 2)
            # idx, idx_neg = random.sample(range(2), 2)
            a, p = random.sample(range(len(indices[idx])), 2)
            n = random.sample(range(len(indices[idx_neg])), 1)[0]
            
            # print("idx: [{}, {}]\na: {}, p: {}, n: {}".format(idx, idx_neg, a, p, n))

            anch = players[idx][a, ..., ..., ...].view(1, 3, SIZE, SIZE).to(device)
            pos = players[idx][p, ..., ..., ...].view(1, 3, SIZE, SIZE).to(device)
            neg = players[idx_neg][n, ..., ..., ...].view(1, 3, SIZE, SIZE).to(device)
                
            imgs = torch.cat([anch, pos, neg], 0)

            imgs = Variable(imgs.to(device))
        
            if model == yolo:
                loss, outputs = model(imgs, targets=0)
                anchor = outputs[0, ...].view(1, outputs.size(1))
                positive = outputs[1, ...].view(1, outputs.size(1))
                negative = outputs[2, ...].view(1, outputs.size(1))
            else:
                # outputs = model(imgs)
                anchor = model(anch)
                positive = model(pos)
                negative = model(neg)
                # anchor = outputs[0, ...].view(1, outputs.size(1))
                # positive = outputs[1, ...].view(1, outputs.size(1))
                # negative = outputs[2, ...].view(1, outputs.size(1))
                loss = loss_fcn(anchor, positive, negative)
            
            loss.backward()

            # if loss.item() > (sum_loss / (epoch+1)):
            #     print("Appending new indices, loss: {:0.4f}, avg_loss {:0.4f}".format(loss.item(), sum_loss / (epoch+1)))
            #     a_indices.append(a)
            #     p_indices.append(p)
            #     n_indices.append(n)

            # if len(a_indices) > max_len:
            #     rand_idx = random.randint(0, len(a_indices)-1)
            #     a_indices.pop(rand_idx)
            #     p_indices.pop(rand_idx)
            #     n_indices.pop(rand_idx)

            # Log progress
            # print("---")
            # dist_ap = torch.dist(anchor, positive).item()
            # dist_an = torch.dist(anchor, negative).item()
            # print("|\tidx: {:03d} --> a: {:03d}, p: {:03d}".format(idx, a, p))
            # print("|\tneg: {:03d} -->         n: {:03d}".format(idx_neg, n))
            # print("|\tD(a,p): {:4.1f}, D(a,n): {:4.1f}, loss: {:7.4f}".format(dist_ap, dist_an, loss.item()))
            # print("---")

            # if (batch_i+1) % BATCH == 0:
                # with torch.no_grad():
                #     for param in model.parameters():
                #         param -= LR * param.grad
                # print("Optimizing...")
                # Accumulates gradient before each step
        optimizer.step()
        optimizer.zero_grad()

        # print("\r\tloss {:.4f}, time: {}".format(loss.item(), epoch_time), end="")
        # print("Epoch {:d}".format(epoch))

        # print("Epoch {:d}\r".format(epoch), end="")
        # visualizePCA(model, players, "PCA_epoch_{:03d}.jpg".format(epoch))

        if (epoch+1) % 20 == 0:
            # print("================================================= vv epoch {:d} vv ===============================================================".format(epoch))
            # testModel(model, player1[..., ..., ..., ...], player2[..., ..., ..., ...])
            # print("================================================= ^^ epoch {:d} ^^ ===============================================================".format(epoch))
            torch.save(model.state_dict(), "checkpoints/yolov3_id_ckpt_%d.pth" % epoch)
            visualizeLDA(model, test_players, "LDA/LDA_epoch_{:03d}.jpg".format(epoch))
        if (epoch) % 50 == 0:
            # print("---")
            # dist_ap = torch.dist(anchor, positive).item()
            # dist_an = torch.dist(anchor, negative).item()
            # print("|\tidx: {:03d} --> a: {:03d}, p: {:03d}".format(idx, a, p))
            # print("|\tneg: {:03d} -->         n: {:03d}".format(idx_neg, n))
            # print("|\tD(a,p): {:4.1f}, D(a,n): {:4.1f}, loss: {:7.4f}".format(dist_ap, dist_an, loss.item()))
            # print("---")
            print("================================================= vv epoch {:d} vv ===============================================================".format(epoch))
            idx1, idx2 = [0, 1]
            num_players = min(8, min(len(indices[idx1]), len(indices[idx2])))
            # testModel(model,
            #     players[idx1][random.sample(range(len(indices[idx1])), num_players), ..., ..., ...],
            #     players[idx2][random.sample(range(len(indices[idx2])), num_players), ..., ..., ...]
            # )
            # testmodelShort(model, players)
            # visualizeLDA(model, players, "LDA/LDA_epoch_{:03d}.jpg".format(epoch))
            # visualizePCA(model, players, "PCA_epoch_{:03d}.jpg".format(epoch))
            
            epoch_time = datetime.timedelta(seconds=time.time() - epoch_time)
            elapsed_time = datetime.timedelta(seconds=time.time() - start_time)
            print("Epoch time: {}".format(epoch_time))
            print("Elapsed time: {}".format(elapsed_time))
            print("================================================= ^^ epoch {:d} ^^ ===============================================================".format(epoch))
            
    # testModel(model, player1, player2)
    
    training_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Training time: {}".format(training_time))

    # testmodelShort(model, players)
    visualizeLDA(model, players, "LDA/LDA_final.jpg")
    # visualizePCA(model, players, "PCA_final.jpg")
    
    print("================================================= Test set ===============================================================")
    # testmodelShort(model, test_players)
    visualizeLDA(model, test_players, "LDA/LDA_test_set.jpg")
    # visualizePCA(model, test_players, "PCA_test_set.jpg")
    print("================================================= Test set ===============================================================")
    
    overall_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Overall time: {}".format(overall_time))