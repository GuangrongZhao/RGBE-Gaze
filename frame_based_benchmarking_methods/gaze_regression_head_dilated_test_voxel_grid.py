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
from model import DilatedNet_voxel_grid

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
    parser.add_argument("--epoch", default=150, type=int, help="The number of epochs")
    parser.add_argument("--network", default="DilatedNet_voxel_grid", help="The type of network")

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
    

    # 首先计算总行数
    total_rows = 0

    for ii in range(train_start,train_end):
        for jj in [1,2,3,4,5,6]:
            
           
            if jj == 5 or jj == 6: 
                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
        

            else:

                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
        

            # #   print(f_data['data'].shape[1])
            #   new_array = np.arange(0,f_data['data'].shape[1], 2)  # 抽取一半的比例以维持2% 水平
            #   total_rows += len(new_array) 
            # #   print(len(new_array) )
            # #   print(total_rows)
            # else:
            total_rows += f_data['data'].shape[1]  # 假设 'data' 数据集在轴1上的大小相同


            print(total_rows)
            # 预分配数组空间
    train_data = np.empty((total_rows, 1, 64, 96))
    train_voxel = np.empty((total_rows, 5, 64, 96))  # 适应您的数据形状
    train_label = np.empty((total_rows, 2))  # 适应您的数据形状
    train_headpose = np.empty((total_rows, 2))  # 适应您的数据形状

    index = 0

    # 填充数据
    for ii in range(train_start,train_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)


            if jj == 5 or jj == 6:
                    f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/64_96_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_voxel = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/64_96_voxel_grid_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_label = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_headpose = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/headpose_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
            else:
                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/64_96_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_voxel = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/64_96_voxel_grid_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_label = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_headpose = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/headpose_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')




            train_data_temp = ((f_data['data'].value).T).reshape(-1, 1, 64, 96)
            train_voxel_temp = ((f_voxel['data'].value)).reshape(-1, 5,64, 96)
            train_label_temp = ((f_label['data'].value).T)
            train_headpose_temp = ((f_headpose['data'].value).T)


            # print(train_data_temp.shape)
            # if jj == 5 or jj == 6: 
            #     new_array = np.arange(0, train_data_temp.shape[0] , 2)
            #     train_data_temp  =  train_data_temp[new_array,:,:,:]
            #     train_label_temp = train_label_temp[new_array,:]
            #     train_voxel_temp = train_voxel_temp[new_array,:,:,:]
            #     train_headpose_temp = train_headpose_temp[new_array,:]

                # print(train_data_temp.shape)


            rows = train_data_temp.shape[0] 
            # print( rows)
            train_data[index:index + rows, :, :, :] = train_data_temp
            train_label[index:index + rows, :] = train_label_temp
            train_voxel[index:index + rows, :, :, :] = train_voxel_temp
            train_headpose[index:index + rows, :] = train_headpose_temp

            index += rows
    print(train_data.shape)
    print(train_label.shape)
    print(train_voxel.shape)


    trainDataset = Data.TensorDataset((torch.from_numpy(train_data).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_voxel).type(torch.FloatTensor)),(torch.from_numpy(train_label).type(torch.FloatTensor)),(torch.from_numpy(train_headpose).type(torch.FloatTensor)))

    n_train = len(trainDataset)
    
    trainloader  = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            # pin_memory=True
        )


    # for ii in range(train_start,train_end):
    #     for jj in [1,2,3,4]:
    #         # print(jj)
    #         f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/64_96_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    #         f_label = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    #         f_head = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/headpose_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

    #        # print(((f_data['data'].value).T).shape)
    #        # print(((f_label['data'].value).T).shape)

    #         train_data_temp = ((f_data['data'].value).T).reshape(-1,1,64,96)
    #         train_label_temp = ((f_label['data'].value).T)
    #         train_head_temp = ((f_head['data'].value).T)
    #         if flag == 1:
    #                 train_data = train_data_temp
    #                 train_label = train_label_temp
    #                 train_head = train_head_temp 
    #                 flag = flag +1
    #         else:
    #                 train_data = np.concatenate((train_data,train_data_temp),axis=0)
    #                 train_label = np.concatenate((train_label,train_label_temp),axis=0)
    #                 train_head = np.concatenate((train_head,train_head_temp),axis=0)
                    
    # print(train_data.shape)
    # print(train_label.shape)
    # print(train_head.shape)

 
    # trainDataset = Data.TensorDataset( (torch.from_numpy(train_data).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_label).type(torch.FloatTensor)),(torch.from_numpy(train_head).type(torch.FloatTensor)))


    # n_train = len(trainDataset)
    
    # trainloader  = torch.utils.data.DataLoader(
    #         trainDataset,
    #         batch_size=64,
    #         shuffle=True,
    #         num_workers=4,
    #         # pin_memory=True
    #     )

####################################### #########################################################################################################

    total_rows = 0

    for ii in range(test_start,test_end):
        for jj in [1,2,3,4,5,6]:
            
           
            if jj == 5 or jj == 6: 
                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
        

            else:

                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
        

            # #   print(f_data['data'].shape[1])
            #   new_array = np.arange(0,f_data['data'].shape[1], 2)  # 抽取一半的比例以维持2% 水平
            #   total_rows += len(new_array) 
            # #   print(len(new_array) )
            # #   print(total_rows)
            # else:
            total_rows += f_data['data'].shape[1]  # 假设 'data' 数据集在轴1上的大小相同


                #  print(total_rows)
            # 预分配数组空间
    test_data = np.empty((total_rows, 1, 64, 96))
    test_voxel = np.empty((total_rows, 5, 64, 96))  # 适应您的数据形状
    test_label = np.empty((total_rows, 2))  # 适应您的数据形状
    test_headpose = np.empty((total_rows, 2))  # 适应您的数据形状

    index = 0

    # 填充数据
    for ii in range(test_start,test_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)


            if jj == 5 or jj == 6:
                    f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/64_96_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_voxel = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/64_96_voxel_grid_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_label = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                    f_headpose = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid_smooth/headpose_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
            else:
                f_data = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/64_96_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_voxel = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/64_96_voxel_grid_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_label = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')
                f_headpose = h5py.File(os.path.join('/home/sduu2/桌面/data_for_voxel_grid/headpose_R_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'), 'r')




            test_data_temp = ((f_data['data'].value).T).reshape(-1, 1, 64, 96)
            test_voxel_temp = ((f_voxel['data'].value)).reshape(-1, 5,64, 96)
            test_label_temp = ((f_label['data'].value).T)
            test_headpose_temp = ((f_headpose['data'].value).T)


            # print(test_data_temp.shape)
            # if jj == 5 or jj == 6: 
            #     new_array = np.arange(0, test_data_temp.shape[0] , 2)
            #     test_data_temp  =  test_data_temp[new_array,:,:,:]
            #     test_label_temp = test_label_temp[new_array,:]
            #     test_voxel_temp = test_voxel_temp[new_array,:,:,:]
            #     test_headpose_temp = test_headpose_temp[new_array,:]

                # print(test_data_temp.shape)


            rows = test_data_temp.shape[0] 
            # print( rows)
            test_data[index:index + rows, :, :, :] = test_data_temp
            test_label[index:index + rows, :] = test_label_temp
            test_voxel[index:index + rows, :, :, :] = test_voxel_temp
            test_headpose[index:index + rows, :] = test_headpose_temp

            index += rows
    print(test_data.shape)
    print(test_label.shape)
    print(test_voxel.shape)


    testDataset = Data.TensorDataset((torch.from_numpy(test_data).type(torch.FloatTensor)/ 255),(torch.from_numpy(test_voxel).type(torch.FloatTensor)),(torch.from_numpy(test_label).type(torch.FloatTensor)),(torch.from_numpy(test_headpose).type(torch.FloatTensor)))

    n_test = len(testDataset)
    
    testloader  = torch.utils.data.DataLoader(
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
    elif args.network == "DilatedNet_voxel_grid":
        model =  DilatedNet_voxel_grid.DilatedNet().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9,0.95))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=  8000, gamma=0.1)
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

    with open(os.path.join("/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/", f"new_total_smooth_pursuit_voxel_grid_histeq_same_parame_no_scale_DilatedNet_first_right"), 'w') as outfile:
     for epoch in tqdm(range(args.epoch)):

            for i, data in enumerate(trainloader):
                model.train()
                x, voxel,labels ,head= data
                x, voxel,labels, head = x.to(device), voxel.to(device), labels.to(device),head.to(device)
            
                pixel_preds =  model(x,voxel,head)

                loss= torch.nn.MSELoss(reduction='mean')(pixel_preds,labels)          
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
                # if args.network == resnet50:
            print(loss)      


            with torch.no_grad(): 

                model.eval()
                Testloss = 0
                Testlosstotal = 0
                Testdistance = 0

                kkk = 1
                for data in testloader:  # 循环每一个batch

                    inputs,  voxel,labels,head  = data
                    inputs,  voxel,labels,head = inputs.to(device), voxel.to(device), labels.to(device),head.to(device)
                    # print(labels.cpu().numpy())
                    label = model(inputs, voxel,head)
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
               torch.save(model.state_dict(), os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/', f"new_total_smooth_pursuit_voxel_grid_Iter_{epoch}_same_parame_no_scale_DilatedNet_first_right.pt"))

