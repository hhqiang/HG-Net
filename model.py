import torch
import math
import torchvision
from torchsummary import  summary
# from torchinfo import summary
from ptflops import get_model_complexity_info


from torch import nn
from efficientnet_pytorch import EfficientNet



class Multinet_mpii(nn.Module):
    def __init__(self, yaw_num_bins, pitch_num_bins):
        super(Multinet_mpii, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3')
        state_dict = torch.load('./model/efficientnet-b3.pth')
        self.model.load_state_dict(state_dict)
        self.out_features = self.model._fc.out_features
        self.fc_yaw = nn.Linear(self.out_features, yaw_num_bins)   # mpiigaze-84 gaze360-180
        self.fc_pitch = nn.Linear(self.out_features, pitch_num_bins)

        # mpiigaze 
        self.fc_yaw_0 = nn.Linear(self.out_features, 28)
        self.fc_yaw_1 = nn.Linear(self.out_features, 12)
        self.fc_yaw_2 = nn.Linear(self.out_features, 4)
        self.fc_yaw_3 = nn.Linear(self.out_features, 2)
        self.fc_pitch_0 = nn.Linear(self.out_features, 28)
        self.fc_pitch_1 = nn.Linear(self.out_features, 12)
        self.fc_pitch_2 = nn.Linear(self.out_features, 4)
        self.fc_pitch_3 = nn.Linear(self.out_features, 2)


    def forward(self,x):
        x = self.model(x)

        # gaze
        pre_yaw =  self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)

        pre_yaw_0 = self.fc_yaw_0(x)
        pre_pitch_0 = self.fc_pitch_0(x)

        pre_yaw_1 = self.fc_yaw_1(x)
        pre_pitch_1 = self.fc_pitch_1(x)
        
        pre_yaw_2 = self.fc_yaw_2(x)
        pre_pitch_2 = self.fc_pitch_2(x)
        
        pre_yaw_3 = self.fc_yaw_3(x)
        pre_pitch_3 = self.fc_pitch_3(x)
        
        return pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3


class Multinet_360(nn.Module):
    def __init__(self, yaw_num_bins, pitch_num_bins):
        super(Multinet_360, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3')
        state_dict = torch.load('./model/efficientnet-b3.pth')
        self.model.load_state_dict(state_dict)
        # torch.nn.init.xavier_uniform_(conv.weight)
        # torch.nn.init.xavier_uniform_(conv.bias, 0)
        self.out_features = self.model._fc.out_features
        self.fc_yaw = nn.Linear(self.out_features, yaw_num_bins)   # mpiigaze-84 gaze360-180
        self.fc_pitch = nn.Linear(self.out_features, pitch_num_bins)


        # gaze360
        self.fc_yaw_0 = nn.Linear(self.out_features, 90)
        self.fc_yaw_1 = nn.Linear(self.out_features, 36)
        self.fc_yaw_2 = nn.Linear(self.out_features, 12)
        self.fc_yaw_3 = nn.Linear(self.out_features, 4)
        
        self.fc_pitch_0 = nn.Linear(self.out_features, 90)
        self.fc_pitch_1 = nn.Linear(self.out_features, 36)
        self.fc_pitch_2 = nn.Linear(self.out_features, 12)
        self.fc_pitch_3 = nn.Linear(self.out_features, 4)




    def forward(self,x):
        x = self.model(x)

        # gaze
        pre_yaw =  self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)

        pre_yaw_0 = self.fc_yaw_0(x)
        pre_pitch_0 = self.fc_pitch_0(x)

        pre_yaw_1 = self.fc_yaw_1(x)
        pre_pitch_1 = self.fc_pitch_1(x)
        
        pre_yaw_2 = self.fc_yaw_2(x)
        pre_pitch_2 = self.fc_pitch_2(x)
        
        pre_yaw_3 = self.fc_yaw_3(x)
        pre_pitch_3 = self.fc_pitch_3(x)
        
        return pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3


class Multinet_eye(nn.Module):
    def __init__(self, yaw_num_bins, pitch_num_bins):
        super(Multinet_eye, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b3')
        state_dict = torch.load('./model/efficientnet-b3.pth')
        self.model.load_state_dict(state_dict)
        self.out_features = self.model._fc.out_features
        self.fc_yaw = nn.Linear(self.out_features, yaw_num_bins)   # mpiigaze-84 gaze360-180
        self.fc_pitch = nn.Linear(self.out_features, pitch_num_bins)

        # mpiigaze 
        self.fc_yaw_0 = nn.Linear(self.out_features, 28)
        self.fc_yaw_1 = nn.Linear(self.out_features, 12)
        self.fc_yaw_2 = nn.Linear(self.out_features, 4)
        self.fc_yaw_3 = nn.Linear(self.out_features, 2)
        self.fc_pitch_0 = nn.Linear(self.out_features, 28)
        self.fc_pitch_1 = nn.Linear(self.out_features, 12)
        self.fc_pitch_2 = nn.Linear(self.out_features, 4)
        self.fc_pitch_3 = nn.Linear(self.out_features, 2)


    def forward(self,x):
        x = self.model(x)

        # gaze
        pre_yaw =  self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)

        pre_yaw_0 = self.fc_yaw_0(x)
        pre_pitch_0 = self.fc_pitch_0(x)

        pre_yaw_1 = self.fc_yaw_1(x)
        pre_pitch_1 = self.fc_pitch_1(x)
        
        pre_yaw_2 = self.fc_yaw_2(x)
        pre_pitch_2 = self.fc_pitch_2(x)
        
        pre_yaw_3 = self.fc_yaw_3(x)
        pre_pitch_3 = self.fc_pitch_3(x)
        
        return pre_pitch, pre_pitch_0, pre_pitch_1, pre_pitch_2, pre_pitch_3, pre_yaw, pre_yaw_0, pre_yaw_1, pre_yaw_2, pre_yaw_3

if __name__=="__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 45).to(device)
    # summary(model,(3,224,224))

    #---------------------------------------------------------------------
    # FLOPs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', FLOPs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    # ---------------------------------------------------------------------
    # X = torch.rand(size=(3, 224, 224), dtype=torch.float32)
    # for layer in model:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

    model = Multinet_mpii(84,84)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)