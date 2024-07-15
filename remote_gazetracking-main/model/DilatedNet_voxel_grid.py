import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary
import torch
import matplotlib.pyplot as plt



class DilatedNet(nn.Module):
    def __init__(self):
        super(DilatedNet, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        eyeconv = models.vgg16(pretrained=True).features
        # self.eyeStreamConv.load_state_dict(vgg16.features[0:9].state_dict())
        # self.faceStreamPretrainedConv.load_state_dict(vgg16.features[0:15].state_dict())

        # Eye Stream, is composed of Conv DConv and FC layers.
        self.eyeStreamConv = nn.Sequential(
            eyeconv[0], #nn.Conv2d(3,   64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            eyeconv[2], #nn.Conv2d(64,  64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            eyeconv[5], #nn.Conv2d(64,  128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            eyeconv[7], #nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.eyeStreamConv[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))


        self.voxeleyeStreamConv = nn.Sequential(
                    eyeconv[0], #nn.Conv2d(3,   64,  3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    eyeconv[2], #nn.Conv2d(64,  64,  3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    eyeconv[5], #nn.Conv2d(64,  128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    eyeconv[7], #nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
        self.voxeleyeStreamConv[0] = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))


        self.eyeStreamDConv = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1,  dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 1, dilation=(4, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, dilation=(5, 11)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )



        self.voxeleyeStreamDConv = nn.Sequential(
                    nn.Conv2d(128, 64, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(64, 64, 3, 1,  dilation=(2, 2)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(64, 128, 3, 1, dilation=(4, 5)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(128, 128, 3, 1, dilation=(5, 11)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )



        self.eyeStreamFC = nn.Sequential(
            nn.Linear(128*4*6, 256),
            # nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )


        self.voxeleyeStreamFC = nn.Sequential(
                    nn.Linear(128*4*6, 256),
                    # nn.Linear(2048, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5)
                )

        self.totalFC = nn.Sequential(
            nn.Linear(256+256+2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x_in,voxel_grid, headpose):


        # print(x_in.shape)

        leftEyeFeature = self.eyeStreamConv(x_in)
        # leftEyeFeature_dconv = leftEyeFeature
        leftEyeFeature = self.eyeStreamDConv(leftEyeFeature)

        # visualize_features(leftEyeFeature, 'Left Eye Feature Maps')
        leftEyeFeature = torch.flatten(leftEyeFeature, start_dim=1)
        leftEyeFeature = self.eyeStreamFC(leftEyeFeature)


        # Get Right feature replace by voxel grid
        rightEyeFeature = self.voxeleyeStreamConv(voxel_grid)
        # rightEyeFeature_dconv = rightEyeFeature
        rightEyeFeature = self.voxeleyeStreamDConv(rightEyeFeature)

        # visualize_features(rightEyeFeature, 'Right Eye Feature Maps')
        rightEyeFeature = torch.flatten(rightEyeFeature, start_dim=1)
        rightEyeFeature = self.voxeleyeStreamFC(rightEyeFeature)





        feature = torch.cat((leftEyeFeature,rightEyeFeature,headpose), 1)
        gaze = self.totalFC(feature)

        # return gaze , leftEyeFeature_dconv, rightEyeFeature_dconv
        return gaze
if __name__ == '__main__':
    m = DilatedNet()
    m.to("cuda")
    # feature = {"face":torch.zeros(64, 3, 96, 96).to("cuda"),
    #             "left":torch.zeros(64, 3, 64,96).to("cuda"),
    #             "right":torch.zeros(64, 3, 64,96).to("cuda")
    #           }
    head = torch.zeros(64, 2).cuda()
    left = torch.zeros(64, 1, 64,96).to("cuda")
    voxel = torch.zeros(64, 5, 64,96).to("cuda")
    a = m(left,voxel,head)
    # a = m(left)
    print(m)

    # input_sizes = [(1, 64, 96), (5, 64, 96), (2,)]
    # summary(m, input_size=input_sizes)
    # # print("Total input size: {:.2f} MB".format(total_input_size))

    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total trainable parameters: {total_params}")
