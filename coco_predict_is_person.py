import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import CocoDataset
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pdb


batch_size = 32
# resolution = (32,32)
resolution = (256,256)

def plot_dataset_sample(dataset):
    fig = plt.figure()
    i = random.randint(0, len(dataset))
    sample = dataset[i]
    print(i, sample['image'].shape, sample['isPerson'], sample['classMask'].shape)

    ax = plt.subplot(2, 1, 1)
    plt.tight_layout()
    ax.set_title("isPerson: " + str(sample['isPerson']))
    ax.axis('off')
    plt.imshow(sample['image'])

    ax = plt.subplot(2, 1, 2)
    ax.axis('off')
    plt.imshow(sample['classMask'])

    plt.show()

def test_plot():
    scale = CocoDataset.Rescale(resolution)
    dataset = CocoDataset.CocoDataset('val2017', transform=scale)
    for _ in range(50):
        plot_dataset_sample(dataset)

def calcAccuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images = data['image']
            labels = data['isPerson']
            outputs = net(images)
            predicted = outputs.data > 0.5
            total += labels.size(0)
            correct += (predicted == labels.byte()).sum().item()

    return correct/total

# test_plot()

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([CocoDataset.Rescale(resolution),
                               CocoDataset.ToTensor(), 
                               CocoDataset.Normalize()])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainset = CocoDataset.CocoDataset('train2017', transform=transform, length=128)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
testset = CocoDataset.CocoDataset('val2017', transform=transform, length=100)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

########################################################################
# Let us show some of the training images, for fun.

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
nxt = dataiter.next()
images = nxt['image']
labels = nxt['isPerson']

# show images
# print(labels)
# imshow(torchvision.utils.make_grid(images))

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_layers = 8
        self.conv1 = nn.Conv2d(3, n_layers, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_layers, n_layers*2, 3, padding=1) 
        self.conv3 = nn.Conv2d(n_layers*2, n_layers*2*2, 3, padding=1) 
        self.conv4 = nn.Conv2d(n_layers*2*2, n_layers*2*2*2, 3, padding=1) 
        self.conv5 = nn.Conv2d(n_layers*2*2*2, n_layers*2*2*2*2, 3, padding=1) 
        sz = int(resolution[0]/2/2/2/2/2)
        self.fc1 = nn.Linear(n_layers*2*2*2*2*sz*sz, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 128
        x = self.pool(F.relu(self.conv2(x))) # 64
        x = self.pool(F.relu(self.conv3(x))) # 32
        x = self.pool(F.relu(self.conv4(x))) # 16
        x = self.pool(F.relu(self.conv5(x))) # 8
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.act(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

print_n = 20
for epoch in range(100):  # loop over the dataset multiple times
    
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    t0 = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs = data['image']
        labels = data['isPerson']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()

        # running train accuracy
        epoch_total += labels.size(0)
        running_total += labels.size(0)
        predicted = outputs.data > 0.5
        correct_count = (predicted == labels.byte()).sum().item()
        epoch_correct += correct_count
        running_correct += correct_count

        if i % print_n == print_n-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f trainAcc: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_n, running_correct/running_total))
            running_loss = 0.0
            running_correct = 0
            running_total = 0

    t1 = time.time()
    # trainAcc = calcAccuracy(net, trainloader)
    trainAcc = epoch_correct / epoch_total
    testAcc = calcAccuracy(net, testloader)
    # testAcc = trainAcc
    t2 = time.time()
    print('[%d] loss: %.5f trainAcc: %.3f testAcc: %.3f trainTime: %.3f accTime: %.3f' %
          (epoch + 1, epoch_loss/(i+1), trainAcc, testAcc, t1 - t0, t2 - t1 ))

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
data = dataiter.next()
images = data['image']
labels = data['isPerson']

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(batch_size)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs, 1)
predicted = outputs > 0.5

print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(batch_size)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

acc = calcAccuracy(net, testloader)
print('Accuracy of the network on the %d test images: %d %%' % (
    len(testset), 100 * acc))




