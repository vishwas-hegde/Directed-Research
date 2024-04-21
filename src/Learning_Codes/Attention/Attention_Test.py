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
        if Index < 180:
            continue
        if Index > 204:
            break
        Im = cv2.cvtColor(cv2.imread(Paths[Index]), cv2.COLOR_BGR2GRAY)
        
        # Divide the image into equal parts of 32x32
        Im = torch.tensor(Im)
        Mod = torch.zeros(64, 1024)
        count = 0
        for i in range(8):
            for j in range(8):
                Mod[count] = torch.flatten(Im[i*32:(i+1)*32, j*32:(j+1)*32]*(count/Mod.shape[0]))
                # cv2.imwrite('/media/storage/lost+found/WPI/Sem2/DR/map_'+str(count)+'.png', Mod[count].numpy())
                count+=1

        Mod = Mod/255

        for i in range(Mod.shape[0]):
            if i == 0:
                Final_Im = Mod[i]
                Final_Im = torch.unsqueeze(Final_Im, 0)
            else:
                Final_Im = torch.cat((Final_Im, torch.unsqueeze(Mod[i],0)))
                # print(Mod[i].shape)
                # print(Mod[i])
        
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

    def __len__(self):
        return len(self.Dataset)

    def __getitem__(self, Index):
        return self.Dataset[Index], torch.tensor(self.Labels[str(Index+180)])


class ViT_Encoder(nn.Module):
    def __init__(self):
        super(ViT_Encoder, self).__init__()
        
        self.Q = nn.Linear(64, 64)
        self.K = nn.Linear(64, 64)
        self.V = nn.Linear(64, 64)
        self.LayerNorm = nn.LayerNorm(128)
        self.Lin = nn.Linear(128, 64)

        self.L1 = nn.Linear(64, 128)
        # self.L2 = nn.Linear(128, 256)
        # self.L3 = nn.Linear(256, 128)
        self.L4 = nn.Linear(128, 64)
        self.L5 = nn.Linear(64, 3)

        self.out = nn.Linear(3072, 3)

        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(0.3)
        

    def forward(self, x):
        
        H1 = self.SelfAttention_Block(x)
        H2 = self.SelfAttention_Block(x)

        x = torch.cat((H1, H2), 1)
        x = self.LayerNorm(x)
        x = self.Dropout(self.ReLU(self.Lin(x)))
        # x = self.Dropout(self.ReLU(self.L1(x)))
        # x = self.Dropout(self.ReLU(self.L2(x)))
        # x = self.Dropout(self.ReLU(self.L3(x)))
        # x = self.Dropout(self.ReLU(self.L4(x)))
        x = self.Dropout(self.ReLU(self.L5(x)))
        x = x.view(1,-1)
        
        x = self.ReLU(self.out(x))
        
        return torch.squeeze(x)

    def SelfAttention_Block(self, x):
        
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        K = torch.transpose(K, 0, 1)
        Out = torch.matmul(Q, K)
        Out = Out/torch.sqrt(torch.tensor(64.0))
        Out = F.softmax(Out, dim=-1)

        Out = torch.matmul(Out, V)

        return Out


if __name__ == '__main__':

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Dataset = PreProcess_data('/media/storage/lost+found/WPI/Sem2/DR/Map/maps_png/')
    Train_Dataset = Vit_dataset(Dataset)
    Train_Dataloader = DataLoader(Train_Dataset, batch_size=1,shuffle=False)
    Model = ViT_Encoder().to(Device)
    Checkpoint = torch.load('/media/storage/lost+found/WPI/Sem2/DR/Attention_ckpt/40_2.670705908569842.pt')
    Model.load_state_dict(Checkpoint)
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=0.0001)
    Loss = nn.MSELoss()

    Model.eval()

    Mean_Loss = []
    Final_Dict = {}
    for i,(Im,Label) in enumerate(Train_Dataloader):
        Optimizer.zero_grad()
        Label = torch.squeeze(Label, 0).to(Device)
        Im = torch.permute(Im, (2,1,0))
        Im = torch.squeeze(Im)
        Pred = Model(Im.to(Device))
        # if Pred[0] > Label[0] and Pred[0] < 9:
        #     Label[0] = Pred[0]
        L = Loss(Pred, Label)
        Mean_Loss.append(L.item())
        print(i)
        print('Pred_L:',Pred)
        print('Actual:',Label)
        print('Loss:',L.item())
        print('##################')
        Temp = Pred.cpu().detach().numpy().tolist()
        Temp[1] = Temp[1]/10
        Final_Dict[str(i)] = Temp 
    Mean_Loss = sum(Mean_Loss)/len(Mean_Loss)
    print(Mean_Loss)
    print(Final_Dict)
    f = open('/media/storage/lost+found/WPI/Sem2/DR/Attention.json', 'w')
    json.dump(Final_Dict, indent=4, fp=f)
    f.close()
        