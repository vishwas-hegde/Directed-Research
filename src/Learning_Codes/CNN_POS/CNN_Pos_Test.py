import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json


# Paths = sorted(glob('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/*.png'))

def PreProcess_data(Im_Dir):

    # Paths = sorted(glob('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/*.png'))
    Paths = sorted(glob(Im_Dir+'*.png'))
    Dataset = []

    for Index in range(len(Paths)):
        # if Index > (0.8*len(Paths)):
        if Index < (160):
            continue
        if Index > (179):
            break
        Im = cv2.cvtColor(cv2.imread(Paths[Index]), cv2.COLOR_BGR2GRAY)
        
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
        Data = self.Labels[str(Index+160)] 
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
    Checkpoint = torch.load('/media/storage/lost+found/WPI/Sem2/DR/Pos_CNN/50_2.395594802964479.pt')
    Model.load_state_dict(Checkpoint)
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=0.001)
    Loss = nn.MSELoss()

    Model.eval()

    Mean_Loss = []
    Final_Dict = {}
    for i,(Im,Label) in enumerate(Train_Dataloader):
        Optimizer.zero_grad()
        Label = torch.squeeze(Label, 0).to(Device)
        Pred = Model(Im.to(Device))
        L = Loss(Pred, Label)
        Mean_Loss.append(L.item())
        print(i)
        print('Pred_L:',Pred)
        print('Actual:',Label)
        print('Loss:',L.item())
        print('##################')
        Temp = Pred.cpu().detach().numpy().tolist()
        Temp[1] = Temp[1]/10
        Temp[2] = round(Temp[2])
        Final_Dict[str(i)] = Temp 
    Mean_Loss = sum(Mean_Loss)/len(Mean_Loss)
    print(Mean_Loss)
    print(Final_Dict)
    f = open('/home/dhrumil/Git/Directed-Research/src/Pos_CNN_Learned.json', 'w')
    json.dump(Final_Dict, indent=4, fp=f)
    f.close()