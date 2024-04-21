import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json

################################################################################################################################
# Change the preprocess function. Create vectors of (1024,1) from the image and then concatenate them to form a (64,1024) tensor
# This tensor will be the input to the model
################################################################################################################################

# Paths = sorted(glob('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/*.png'))

def PreProcess_data(Im_Dir):

    # Paths = sorted(glob('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/*.png'))
    Paths = sorted(glob(Im_Dir+'*.png'))
    Dataset = []

    for Index in range(len(Paths)):
        # if Index > (0.8*len(Paths)):
        if Index > 159:
            break
        Im = cv2.cvtColor(cv2.imread(Paths[Index]), cv2.COLOR_BGR2GRAY)
        # print(Paths[Index])
        # Divide the image into equal parts of 32x32
        Im = torch.tensor(Im)
        # Mod = torch.zeros(64, 1024)
        count = 0
        for i in range(8):
            for j in range(8):
                Im[i*32:(i+1)*32, j*32:(j+1)*32] = Im[i*32:(i+1)*32, j*32:(j+1)*32]*(count/64)
                # cv2.imwrite('/media/storage/lost+found/WPI/Sem2/DR/map_'+str(count)+'.png', Mod[count].numpy())
                count+=1
        # cv2.imwrite('/media/storage/lost+found/WPI/Sem2/DR/map_'+str(Index)+'_mod.png', Im.numpy())
        Final_Im = Im/255
        
        Dataset.append(Final_Im)
        # cv2.imwrite('/media/storage/lost+found/WPI/Sem2/DR/map_'+str(Index)+'_mod.png', Final_Im.numpy()*255)

    return Dataset



class Vit_dataset(Dataset):
    def __init__(self, Dataset):
        self.Dataset = Dataset
        F = open('/home/dhrumil/Git/Directed-Research/src/PSO_Output/PSO_Final.json', 'r')
        self.Labels = json.load(F)
        for key in self.Labels.keys():
            self.Labels[key][1] = self.Labels[key][1]*10
            # self.Labels[key][2] = self.Labels[key][2]/10

    def __len__(self):
        return len(self.Dataset)

    def __getitem__(self, Index):
        Data = self.Labels[str(Index)] 
        return self.Dataset[Index], torch.tensor(Data)


class ViT_Encoder(nn.Module):
    def __init__(self):
        super(ViT_Encoder, self).__init__()
        
        self.L1 = nn.Linear(64, 128)
        self.L2 = nn.Linear(128, 256)
        self.L3 = nn.Linear(256, 128)
        self.L4 = nn.Linear(128, 64)
        self.L5 = nn.Linear(64, 3)

        # self.conv0 = nn.Conv2d(1, 8, 3, 1)
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1)
        self.conv5 = nn.Conv2d(16, 4, 3, 1)
        self.conv6 = nn.Conv2d(4, 2, 3, 1)
        self.out1 = nn.Linear(7688, 64)
        self.out2 = nn.Linear(64, 3)

        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(0.2)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(8)

    def forward(self, x):
        # 0-5-6-O1-O2

        # x = self.MaxPool(self.ReLU(self.conv0(x)))
        x = self.MaxPool(self.ReLU(self.conv1(x)))
        # x = self.MaxPool(self.ReLU(self.conv2(x)))
        # x = self.MaxPool(self.ReLU(self.conv3(x)))
        # x = self.MaxPool(self.ReLU(self.conv4(x)))
        # x = self.Dropout(self.MaxPool(self.ReLU(self.conv5(x))))
        x = self.MaxPool(self.ReLU(self.conv6(x)))

        x = x.view(1,-1)
        x = self.ReLU(self.out1(x))
        x = self.out2(x)

        return torch.squeeze(x)

if __name__ == '__main__':

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Dataset = PreProcess_data('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/')
    Train_Dataset = Vit_dataset(Dataset)
    Train_Dataloader = DataLoader(Train_Dataset, batch_size=1,shuffle=False)
    Model = ViT_Encoder().to(Device)
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=0.0001)
    Loss = nn.MSELoss()
    Loss_List = []
    for i in range(300):
        Epoch_Loss = []
        for Im,Label in Train_Dataloader:
            Label = torch.squeeze(Label.to(Device))
            Optimizer.zero_grad()
            Pred = Model(Im.to(Device))
            # Temp_P = Pred.clone()
            # Temp_P[2] = int(Temp_P[2])
            # Temp_L = Label.clone()
            # if Pred[0] > Label[0] and Pred[0] < 0.9:
            #     Temp_L[0] = Pred[0]
            L = Loss(Pred, Label)
            L.backward()
            Optimizer.step()
            Epoch_Loss.append(L.item())
        print('Epoch:', i, 'Loss:', sum(Epoch_Loss)/len(Epoch_Loss))
        # if i > 0 and sum(Epoch_Loss)/len(Epoch_Loss) < min(Loss_List):
        #     torch.save(Model.state_dict(), '/media/storage/lost+found/WPI/Sem2/DR/Pos_CNN/'+str(i)+'_'+str(sum(Epoch_Loss)/len(Epoch_Loss))+'.pt')
        #     break
        Loss_List.append(sum(Epoch_Loss)/len(Epoch_Loss))
        if i%10 == 0:
            print('############################################')
            torch.save(Model.state_dict(), '/media/storage/lost+found/WPI/Sem2/DR/Pos_CNN/'+str(i)+'_'+str(sum(Epoch_Loss)/len(Epoch_Loss))+'.pt')