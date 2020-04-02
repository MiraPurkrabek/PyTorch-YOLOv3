import torch
import random
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from train_siamese_yolo import visualizeLDA

EPOCHS = 100 # Number of epochs
N, D_in = 5, 1024 # Batch size, input dimension
H1, H2, H3, H4, H5, H6 = 1024, 512, 512, 512, 256, 256 # Dimensions of hidden layers
D_out = 128 # Output dimension
DEVICE = 'cuda'

selected_IDs = [
    [2, 61, 129, 349], 
    [3, 74, 101, 141, 188],
    [4, 70, 136, 209, 271, 313, 334],
    [10, 146, 192, 261],
    [12, 60, 265, 327]
]

# Load players vectors
vectors = torch.load("output/vectors.pt").to('cpu')
print("Vectors size:", vectors.size())
# print("Vectors:", vectors)

# Load players IDs
ID_list = []
with open("output/IDs.txt", "r") as f:
    for line in f.readlines():
        for word in line.split():
            ID_list.append(int(word))
f.close()
# print("Number of IDs:", len(ID_list))
IDs = torch.Tensor(ID_list)
# print("IDs size:", IDs.size())

players = []
limits = []
# Extract IDs for selected players
for i in range(5):
    idx = torch.BoolTensor(IDs.nelement()).fill_(False)
    for num in selected_IDs[i]:
        idx = (IDs == num) | idx
    p = vectors[..., idx].transpose(1, 0).type(torch.cuda.FloatTensor)
    players.append(p)
    limits.append(int(p.size(0)))

# p2_idx = torch.BoolTensor(IDs.nelement()).fill_(False)
# p3_idx = torch.BoolTensor(IDs.nelement()).fill_(False)
# for num in player1_IDs:
#     p1_idx = (IDs == num) | p1_idx
# for num in player2_IDs:
#     p2_idx = (IDs == num) | p2_idx
# for num in player3_IDs:
#     p3_idx = (IDs == num) | p3_idx
# p1_num = int(torch.sum(p1_idx))
# p2_num = int(torch.sum(p2_idx))
# p3_num = int(torch.sum(p3_idx))
# limits = [p1_num, p2_num, p3_num]
# # print("Player1 examples: {:d}, player2 examples {:d}".format(p1_num, p2_num))

# player1 = vectors[..., p1_idx].transpose(1, 0)
# player2 = vectors[..., p2_idx].transpose(1, 0)
# player3 = vectors[..., p3_idx].transpose(1, 0)
# players.append(player1)
# players.append(player2)
# players.append(player3)
# players.append(torch.nn.functional.normalize(player1, p=1, dim=1))
# players.append(torch.nn.functional.normalize(player2, p=1, dim=1))
# players.append(torch.nn.functional.normalize(player3, p=1, dim=1))
# print("Player 1:", player1.size(), "\nplayer 2:", player2.size(), "\nplayer 3:", player3.size())
# print("Player 1 vec stats:", players[0][0, ...].min(), players[0][0, ...].mean(), players[0][0, ...].max())
# players[0] = torch.nn.functional.normalize(players[0], p=1, dim=1)
# print("Player 1 vec stats:", players[0][0, ...].min(), players[0][0, ...].mean(), players[0][0, ...].max())

idx = 1
a = random.randint(0, limits[idx]-1)
p = random.randint(0, limits[idx]-1)
n = random.randint(0, limits[1-idx]-1)
anchor_old = players[idx][a, ...].view(1, 1024)
positive_old = players[idx][p, ...].view(1, 1024)
negative_old = players[1-idx][n, ...].view(1, 1024)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H5),
    torch.nn.ReLU(),
    torch.nn.Linear(H5, H6),
    torch.nn.ReLU(),
    torch.nn.Linear(H6, D_out),
    torch.nn.ReLU(),
)

# print(model)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=20, p=2.0, eps=1e-02, swap=False, size_average=None, reduce=None, reduction='none')
optimizer = torch.optim.Adam(model.parameters())

model.to(DEVICE)
learning_rate = 1e-2
visualizeLDA(model, players, "LDA/LDA_vectors_before.jpg")
for t in range(EPOCHS):
    model.train()    
    for _ in range(N):
        # Select vector from each player for this epoch
        idx, idx_neg = random.sample(range(len(players)), 2)
        # idx, idx_neg = random.sample(range(2), 2)
        a, p = random.sample(range(limits[idx]), 2)
        n = random.sample(range(limits[idx_neg]), 1)[0]
                
        # print("\tidx: {:03d} --> a: {:03d}, p: {:03d}".format(idx, a, p))
        # print("\tneg: {:03d} -->         n: {:03d}".format(idx_neg, n))
            
        anchor = players[idx][a, ...].view(1, 1024).to(DEVICE)
        positive = players[idx][p, ...].view(1, 1024).to(DEVICE)
        negative = players[idx_neg][n, ...].view(1, 1024).to(DEVICE)
        
        a_pred = model(anchor)
        p_pred = model(positive)
        n_pred = model(negative)

        loss = triplet_loss_fn(a_pred, p_pred, n_pred)
        
        loss.backward()
        print("Epoch {:d} ---> loss {:6f}".format(t, loss.item()))
    
    optimizer.step()
    optimizer.zero_grad()
    # if t % 10 == 0:

print("Learning done!")
old_dist_ap = torch.dist(anchor_old, positive_old).item()
old_dist_an = torch.dist(anchor_old, negative_old).item()
old_dist_pn = torch.dist(positive_old, negative_old).item()

anchor = model(anchor_old)
positive = model(positive_old)
negative = model(negative_old)

dist_ap = torch.dist(anchor, positive).item()
dist_an = torch.dist(anchor, negative).item()
dist_pn = torch.dist(positive, negative).item()

print("Old distances:\n\tap: {:.4f}, an: {:.4f}, pn: {:.4f}".format(old_dist_ap, old_dist_an, old_dist_pn))
print("New distances:\n\tap: {:.4f}, an: {:.4f}, pn: {:.4f}".format(dist_ap, dist_an, dist_pn))

######################################################################################
####################################### LDA ##########################################
######################################################################################

visualizeLDA(model, players, "LDA/LDA_vectors_final.jpg")

######################################################################################
############################ Logistic regression #####################################
######################################################################################


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.linear_model import LogisticRegression

# # make 3-class dataset for classification
# centers = [[-5, 0, 2], [0, 1.5, 8], [5, -1, 3]]
# X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
# transformation = [[0.4, 0.2, 0.1], [-0.4, 1.2, 1.4], [0.4, -1.2, -0.7]]
# X = np.dot(X, transformation)
    
# num = [p.size(0) for p in players]
# X = torch.cat([model(p).cpu().detach() for p in players]).numpy()
    
# y = []
# for i, n in enumerate(num):
#     y += [i] * n
# y = np.array(y)

# print("X size:", X.shape)
# print("y size:", y.shape)

# plt.clf()
# for multi_class in ('multinomial', 'ovr'):
#     idx = random.sample(range(X.shape[0]), X.shape[0])
#     clf = LogisticRegression(solver='lbfgs', multi_class=multi_class).fit(X[idx, ...], y[idx])

#     tmp = clf.predict_proba(X)
#     # for i in range(y.size):
#     #     print("\t{} - {:d}".format(tmp[i], y[i]))
#     # print the training scores
#     print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

    # create a mesh to plot in
    # h = .02  # step size in the mesh
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#     plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
#     plt.axis('tight')

#     # Plot also the training points
#     colors = "bry"
#     for i, color in zip(clf.classes_, colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
#                     edgecolor='black', s=20)

#     # Plot the three one-against-all classifiers
#     xmin, xmax = plt.xlim()
#     ymin, ymax = plt.ylim()
#     coef = clf.coef_
#     intercept = clf.intercept_

#     def plot_hyperplane(c, color):
#         def line(x0):
#             return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
#         plt.plot([xmin, xmax], [line(xmin), line(xmax)],
#                  ls="--", color=color)

#     for i, color in zip(clf.classes_, colors):
#         plot_hyperplane(i, color)

# plt.savefig("LDA/Linear_reg_test.jpg", format="jpg")


