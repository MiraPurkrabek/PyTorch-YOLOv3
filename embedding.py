import torch
import random

EPOCHS = 500 # Number of epochs
N, D_in = 1, 1024 # Batch size, input dimension
H1, H2, H3, H4, H5, H6 = 1024, 512, 512, 512, 256, 256 # Dimensions of hidden layers
D_out = 128 # Output dimension

player1_IDs = [2, 61, 128, 346]
player2_IDs = [3, 73, 100, 140, 187]
player3_IDs = [46, 86, 121, 164, 284]

# Load players vectors
vectors = torch.load("output/vectors.pt").to('cpu')
# print("Vectors size:", vectors.size())
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

# Extract IDs for selected players
p1_idx = torch.BoolTensor(IDs.nelement()).fill_(False)
p2_idx = torch.BoolTensor(IDs.nelement()).fill_(False)
for num in player1_IDs:
    p1_idx = (IDs == num) | p1_idx
for num in player2_IDs:
    p2_idx = (IDs == num) | p2_idx
p1_num = int(torch.sum(p1_idx))
p2_num = int(torch.sum(p2_idx))
limits = [p1_num, p2_num]
# print("Player1 examples: {:d}, player2 examples {:d}".format(p1_num, p2_num))

player1 = vectors[..., p1_idx]
player2 = vectors[..., p2_idx]
print("Player 1:", player1.size(), "player 2:", player2.size())

anchor = torch.randn(N, D_in)
positive = torch.randn(N, D_in)
negative = torch.randn(N, D_in)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ELU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ELU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ELU(),
    torch.nn.Linear(H3, H4),
    torch.nn.ELU(),
    torch.nn.Linear(H4, H5),
    torch.nn.ELU(),
    torch.nn.Linear(H5, H6),
    torch.nn.ELU(),
    torch.nn.Linear(H6, D_out),
)

# print(model)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='none')

learning_rate = 1e-4
for t in range(EPOCHS):    
    # Select vector from each player for this epoch
    idx = random.randint(0, 1)
    a = random.randint(0, limits[idx]-1)
    p = random.randint(0, limits[idx]-1)
    n = random.randint(0, limits[1-idx]-1)
    
    # idx = 1
    # a = 22
    # p = 100
    # n = 14

    # print("idx: {:d}\ta: {:d}, p: {:d}, n: {:d}".format(idx, a, p, n))
    if idx < 1:
        anchor = player1[..., a].view(1, 1024)
        positive = player1[..., p].view(1, 1024)
        negative = player2[..., n].view(1, 1024)
    else:
        anchor = player2[..., a].view(1, 1024)
        positive = player2[..., p].view(1, 1024)
        negative = player1[..., n].view(1, 1024)

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    a_pred = model(anchor)
    p_pred = model(positive)
    n_pred = model(negative)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = triplet_loss_fn(a_pred, p_pred, n_pred)
    if t % 10 == 0:
        print("Epoch {:d} ---> loss {:6f}".format(t, loss.item()))

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad