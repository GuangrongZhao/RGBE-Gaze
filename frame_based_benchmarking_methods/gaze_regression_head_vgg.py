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

from model import gaitcnn
from model import mpgmodel
from model import resnet50
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

    # cudnn.benchmark=True
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--testuser", default="0", help="The GPU ID")
    parser.add_argument("--cuda", default="2", help="The GPU ID")
    parser.add_argument("--epoch", default=200, type=int, help="The number of epochs")
    parser.add_argument("--network", default="mpgmodel", help="The type of network")

    args = parser.parse_args()

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    print(device)


    #################################### Configure data loader#################################################
    
    train_start = 1
    train_end = 34

    test_start = 34
    test_end = 67
    train_data = []
    train_label = []
    train_head = []
    flag = 1
    
    
    # for ii in [1,2,3]:
    #     for jj in [1,3]:
    #         print(jj)
    #         f_data = h5py.File(os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/remote_dataset/data_for_gaze_network/down_36_60_left_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    #         f_label = h5py.File(os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/remote_dataset/data_for_gaze_network/L_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
      

    #        # print(((f_data['data'].value).T).shape)
    #        # print(((f_label['data'].value).T).shape)

    #         train_data_temp = ((f_data['data'].value).T).reshape(-1,1, 36,60)
    #         train_label_temp = ((f_label['data'].value).T)
    #         if flag == 1:
    #                 train_data = train_data_temp
    #                 train_label = train_label_temp
    #                 flag = flag +1
    #         else:
    #                 train_data = np.concatenate((train_data,train_data_temp),axis=0)
    #                 train_label = np.concatenate((train_label,train_label_temp),axis=0)
                    
                    
    for ii in range(train_start,train_end):
        for jj in [1,2,3,4,5,6]:
            # print(jj)
            f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/R_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_head = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/R_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

           # print(((f_data['data'].value).T).shape)
           # print(((f_label['data'].value).T).shape)

            train_data_temp = ((f_data['data'].value).T).reshape(-1,1,36,60)
            train_label_temp = ((f_label['data'].value).T)
            train_head_temp = ((f_head['data'].value).T)
            if flag == 1:
                    train_data = train_data_temp
                    train_label = train_label_temp
                    train_head = train_head_temp 
                    flag = flag +1
            else:
                    train_data = np.concatenate((train_data,train_data_temp),axis=0)
                    train_label = np.concatenate((train_label,train_label_temp),axis=0)
                    train_head = np.concatenate((train_head,train_head_temp),axis=0)
                    
    print(train_data.shape)
    print(train_label.shape)
    print(train_head.shape)

 
    trainDataset = Data.TensorDataset( (torch.from_numpy(train_data).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_label).type(torch.FloatTensor)),(torch.from_numpy(train_head).type(torch.FloatTensor)))


    n_train = len(trainDataset)
    
    trainloader  = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            # pin_memory=True
        )

####################################### #########################################################################################################

    test_data = []
    test_label = []
    test_head = []
    flag = 1
    # for ii in [1,2,3]:
    #      for jj in [2,4]:
                                            
    #             f_data = h5py.File(os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/remote_dataset/data_for_gaze_network/down_36_60_left_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    #             f_label = h5py.File(os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/remote_dataset/data_for_gaze_network/L_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    #             # for key in f.keys():
    #             #     print(f[key].name)

    #             # print(((f['data'].value).T).shape)
    #             # print(((f['label'].value).T).shape)

    #             test_data_temp = ((f_data['data'].value).T).reshape(-1,1,36,60)
    #             test_label_temp = ((f_label['data'].value).T)
    #             if flag == 1:
                    
    #                     test_data = test_data_temp
    #                     test_label = test_label_temp
    #                     flag = flag +1
    #             else:
    #                     test_data = np.concatenate((test_data,test_data_temp),axis=0)
    #                     test_label = np.concatenate((test_label,test_label_temp),axis=0) 
                        
    for ii in range(test_start,test_end):
         for jj in [1,2,3,4,5,6]:
                                            
                f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/R_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                f_head = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/R_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

                
                # for key in f.keys():
                #     print(f[key].name)

                # print(((f['data'].value).T).shape)
                # print(((f['label'].value).T).shape)

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


    if args.network == "gaitcnn":
        model =  gaitcnn.Net(1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
    elif args.network == "mpgmodel":
        model =  mpgmodel.model().to(device)
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(),lr=0.00001, betas=(0.9,0.95))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5000, gamma=0.1)


    elif args.network == "resnet50":
        model =  resnet50.gaze_network().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)  # goal: maximize Dice score
    # state_dict = torch.load('net_params2.pth')
    # model.load_state_dict(state_dict)

    print(model)
    # model.apply(_init_weights)
    # summary(model, input_size=(1, 36, 60).to(device))

    # optimizer = torch.optim.Adam(model.parameters(),lr=0.00001, betas=(0.9,0.95))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    length = len(trainloader)

    with open(os.path.join("/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/", f"no_scale_MPIIGazempgmodelvgg_first_right"), 'w') as outfile:
        for epoch in tqdm(range(args.epoch)):

            for i, data in enumerate(trainloader):
                model.train()
                x, labels ,head= data
                x, labels, head = x.to(device), labels.to(device),head.to(device)
            
                pixel_preds =  model(x,head )

                loss= torch.nn.MSELoss(reduction='mean')(pixel_preds,labels)          
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if args.network == resnet50:
                scheduler.step()
            print(loss)


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
                    Testloss += torch.nn.MSELoss(reduction='mean')(label, labels)
                    # output = nn.PairwiseDistance(p=2)(label,labels)
        
                    output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))
                    # print(output) # 
                    # print(torch.mean(output)) # 
                    Testdistance += np.mean(output)
                    # Testdistance += output
                    Testlosstotal += 1
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

            if epoch % 10== 0:
               torch.save(model.state_dict(), os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/', f"Iter_{epoch}_no_scale_MPIIGazempgmodelvgg_first_right.pt"))

