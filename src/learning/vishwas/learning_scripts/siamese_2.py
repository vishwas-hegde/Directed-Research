### Resnet18 Siamese Network

from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import random
import os


class SiameseNetwork(nn.Module):
    """
        
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Create a CNN model

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=10),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, kernel_size=7),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 128, kernel_size=4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, kernel_size=4),
        #     nn.ReLU(inplace=True),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),  # Reduced channels and kernel size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),  # Reduced channels and kernel size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3),  # Reduced channels and kernel size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),  # Reduced channels and kernel size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),  # Reduced channels and kernel size
            nn.ReLU(inplace=True),
        )

        # add linear layers to compare between the features of the two images
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 1)


        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        # self.resnet.apply(self.init_weights)
        self.fc1.apply(self.init_weights)
        self.fc2.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x, print_=False):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.fc1(output)

        # Normalize the output
        output = F.normalize(output, p=2, dim=1)

        if print_:
            print(output)
        # print(output.shape)
        output = self.fc2(output)
        output = self.sigmoid(output)
        # print(output)
        # exit()
        if print_:
            print(output)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        # output = torch.cat((output1, output2), 1)

        # # pass the concatenation to the linear layers
        # output = self.fc(output)

        # # pass the out of the linear layers to sigmoid layer
        # output = self.sigmoid(output)
        
        return output1, output2

class APP_MATCHER(Dataset):
    def __init__(self, root_dir, path_file_dir, data, transform=None, random_aug=False):
        super(APP_MATCHER, self).__init__()

        self.root_dir = root_dir

        self.data = data
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1_file = Image.open(img1)
        img2_file = Image.open(img2)
        # change image to gray-scale
        img1_file = img1_file.convert('L')
        img2_file = img2_file.convert('L')
        label = torch.tensor(label, dtype=torch.float)
        if self.transform:
            img1_file = self.transform(img1_file)
            img2_file = self.transform(img2_file)
        return (img1_file, img2_file, label)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # criterion = nn.BCELoss()
    criterion = ContrastiveLoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        output1, output2  = model(images_1, images_2)#.squeeze()
        # print(targets)
        loss = criterion(output1, output2, targets)
        loss.backward()
        optimizer.step()
        # exit()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            output1, output2 = model(images_1, images_2)#.squeeze()
            test_loss += criterion(output1, output2, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop((112, 112)),
        T.ToTensor(),
    ])
    # transform = T.Compose([
    #     T.Resize((224, 224)),
    #     T.ToTensor(),
    # ])

    path_file_dir = 'kinematic/learning_dataset/datapath.txt'
    path_file = open(path_file_dir, 'r')
    data = []
    for line in path_file:
        line = line.strip()
        img1, img2, label = line.split(' ')
        label = int(label)
        data.append((img1, img2, label))
    path_file.close()

    random.shuffle(data)

    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    train_dataset = APP_MATCHER('../data', 'kinematic/learning_dataset/datapath1.txt', train_data, transform=transform)
    test_dataset = APP_MATCHER('../data', 'kinematic/learning_dataset/datapath1.txt', test_data, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=15, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    model = SiameseNetwork().to(device)
    # print summary
    print(args.lr)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 5):#args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    torch.save(model.state_dict(), "siamese_network.pt")

    ## create a txt file containing all the embeddings of the images
    ## and their corresponding labels
    ## this file will be used to comapare the embeddings of the images

    # list the images in the directory
    img_paths = os.listdir('kinematic/maps_png/Train/')
    #sort the images
    img_paths.sort()
    # print(img_paths)
    img_data = []
    for img_path in img_paths:
        img = Image.open(f'kinematic/maps_png/Train/{img_path}')
        img = img.convert('L')
        img = transform(img)
        # add extra dimension to the image to compensate for the batch size
        img = img.unsqueeze(0)
        img_data.append(img)

    params = []
    with open('kinematic/learning_dataset/embeddings_without_labels.txt', 'w') as f:
        for img in img_data:
            img = img.to(device)

            output = model.forward_once(img, print_=False)
            params.append(output.detach().cpu().numpy())
            f.write(f"{output.detach().cpu().numpy()}\n")

    # Testing
    similar_images = []
    test_images = os.listdir('kinematic/maps_png/Test/')
    test_images.sort()
    # get tuple list
    with open('kinematic/learning_dataset/tuplelist.txt', 'r') as f:
        data = f.readlines()
        tuple_data = [tuple(map(int, x.strip().split())) for x in data]
    
    # use tuple_data to compute the accuracy of the model
    for i in range(len(test_images)):
        img = Image.open(f'kinematic/maps_png/Test/{test_images[i]}')
        img = img.convert('L')
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        output = model.forward_once(img, print_=False)
        output = output.detach().cpu().numpy()

        # comapre output with params for least norm
        min_norm = 1000000
        min_idx = -1
        for j in range(len(params)):
            norm = np.linalg.norm(output - params[j])
            if norm < min_norm:
                min_norm = norm
                min_idx = j
        similar_images.append((min_idx, i+51, 1))


        print(f"Image {i+51} is most similar to Image {min_idx}")
    # print(similar_images)
    # print(tuple_data)
    # compute the accuracy
    correct = 0
    for i in range(len(similar_images)):
        if similar_images[i] in tuple_data:
            correct += 1
    print(f"Accuracy: {correct/len(similar_images)}")
    print(f"Correct: {correct}")


def test_dataloader():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_dataset = APP_MATCHER('../data', 'kinematic/learning_dataset/datapath.txt', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        print(images_1.shape, images_2.shape, targets)
        break



if __name__ == '__main__':
    # main()
    # test_dataloader()
    main()

