import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from tqdm import tqdm
import os
import sys
import argparse
import torchsummary
from torchsummary import summary
import copy
sys.path.append("..")
# print(sys.path)
import torch.backends.cudnn as cudnn

from model import vggnet
from model import minist
from model import DilatedNet

from model import mpgmodel

if __name__ == "__main__":
    def pitchyaw_to_vector(pitchyaws):
        r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

        Args:
            pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

        Returns:
            :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
        """
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out


    def vector_to_pitchyaw(vectors):
        r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

        Args:
            vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

        Returns:
            :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
        """
        n = vectors.shape[0]
        out = np.empty((n, 2))
        vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
        out[:, 0] = np.arcsin(vectors[:, 1])  # theta
        out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
        return out


    def angular_error(a, b):
        """Calculate angular error (via cosine similarity)."""
        # print(a.shape)
        a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
        # print(a[0])
        b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
        # print(b[0])
        ab = np.sum(np.multiply(a, b), axis=1)
        # print(ab)
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)

        # Avoid zero-values (to avoid NaNs)
        a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
        b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

        similarity = np.divide(ab, np.multiply(a_norm, b_norm))
        # print(np.arccos(similarity) * 180.0 / np.pi)
        return np.arccos(similarity) * 180.0 / np.pi

    cudnn.benchmark=True
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--testuser", default="0", help="The GPU ID")
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=2, type=int, help="The number of epochs")
    parser.add_argument("--network", default="alexnet", help="The type of network")

    args = parser.parse_args()

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    print(device)


    #################################### Configure data loader#################################################
    userlist = [u for u in range(34,67)]
    # model =  DilatedNet.DilatedNet().to(device)
    # model =  mpgmodel.model().to(device)
    model  =  minist.model().to(device)
    # state_dict = torch.load('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/new_Iter_10_same_parame_no_scale_DilatedNet_first_right.pt')
    state_dict = torch.load('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/inpaper_Iter_10_no_scale_MPIIGazempgmodelminist_first_left.pt')
    # state_dict = torch.load('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/Iter_50_no_scale_MPIIGazempgmodelvgg_first_right.pt')

    model.load_state_dict(state_dict)   
    for user in userlist:
        print(user)  
        
        test_data = []
        test_label = []
        flag = 1
    
        for ii in range(user,user+1):
            for jj in [1,2,3,4,5,6]:
                                                
                    # f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/L_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                    # f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_L_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                    # f_head = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/L_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f


                    f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/L_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                    f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_L_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                    f_head = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/L_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f


                    test_data_temp = ((f_data['data'].value).T).reshape(-1,1,36,60)
                    test_label_temp = ((f_label['data'].value).T)
                    test_head_temp = ((f_head['data'].value).T)
                    if flag == 1:
                        
                            test_data = test_data_temp
                            test_label = test_label_temp
                            test_head = test_head_temp
                            flag = flag +1
                    else:
                            test_data = np.concatenate((test_data,test_data_temp),axis=0)
                            test_label = np.concatenate((test_label,test_label_temp),axis=0)                 
                            test_head = np.concatenate((test_head,test_head_temp),axis=0) 

        print(test_data.shape)
        print(test_label.shape)
        print(test_head.shape)
        
        testDataset = Data.TensorDataset( (torch.from_numpy(test_data).type(torch.FloatTensor)/255),torch.from_numpy(test_label).type(torch.FloatTensor),torch.from_numpy(test_head).type(torch.FloatTensor))
        n_val = len(testDataset)
        testloader = torch.utils.data.DataLoader( 
                testDataset,
                batch_size=512,
                shuffle=False,
                num_workers=8,
                # pin_memory=True
            )
    #################################### Configure data loader#################################################


        # if args.network == "gaitcnn":
        #     model =  gaitcnn.Net(1).to(device)
        #     optimizer = torch.optim.Adam(model.parameters())
        # elif args.network == "minist":
       
        # optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9,0.95))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5000, gamma=0.1)

        # elif args.network == "resnet50":
        #     model =  resnet50.gaze_network().to(device)
        #     optimizer = torch.optim.Adam(model.parameters())
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)  # goal: maximize Dice score
        # state_dict = torch.load('net_params2.pth')
        # model.load_state_dict(state_dict)

        # print(model)
        # model.apply(_init_weights)
        # summary(model, input_size=(1, 36, 60).to(device))

        # optimizer = torch.optim.Adam(model.parameters(),lr=0.00001, betas=(0.9,0.95))
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
        # length = len(trainloader)


        
       
        # model =  mpgmodel.model().to(device)

        with open(os.path.join("/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/ui_minist/user_" + str(user)+ "L"), 'w') as outfile:
            for epoch in tqdm(range(args.epoch)):

               
               

                with torch.no_grad():

                    model.eval()
                    Testloss = 0
                    Testlosstotal = 0
                    Testdistance = 0

                    kkk = 1
                    for data in testloader:  # 循环每一个batch

                        inputs, labels,head  = data
                        inputs, labels,head = inputs.to(device), labels.to(device),head.to(device)
                        # print(labels.cpu().numpy())
                        label = model(inputs,head )
                        # print(label.cpu().numpy()[0])
                        # 取得分最高的那个类
                        Testloss += torch.nn.MSELoss(reduction='sum')(label, labels)
                        # output = nn.PairwiseDistance(p=2)(label,labels)
            
                        output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))
                        # print(output) # 
                        # print(torch.mean(output)) # 
                        Testdistance += np.sum((output))
                        # Testdistance += output
                        Testlosstotal += (output.squeeze().shape[0])
                        kkk = kkk+1

                    # model.train()
                    # torch.save(model.state_dict(), 'net_params4.pth')
                    # print(Testlosstotal)
                    # print('--TestLoss:%6.3f' % (Testloss / Testlosstotal))
                    # print('--Testdistance:%6.3f' % (Testdistance / Testlosstotal))
    # [{i}/{length}
                    # resttime = (timeend - timebegin)/cur * (total-cur)/3600
                    log = f"[{epoch}/{args.epoch}]: loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush() 

                # if epoch % 100== 0:
                    # torch.save(model.state_dict(), os.path.join("/home/sduu2/userspace/zgr/remote_eyetracking/python_code/train_gaze_network/test_result_923/" + str(user)+ "L"+"epoch"+str(epoch)))

