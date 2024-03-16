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
from model import full_face
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
    parser.add_argument("--cuda", default="1", help="The GPU ID")
    parser.add_argument("--epoch", default=50, type=int, help="The number of epochs")
    parser.add_argument("--network", default="full_face", help="The type of network")

    args = parser.parse_args()

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    print(device)


    #################################### Configure data loader#################################################
    
      
    train_start = 1
    train_end = 34

    test_start = 34
    test_end = 67

        # 首先计算总行数
    total_rows = 0

    for ii in range(train_start,train_end):
        for jj in [1,2,3,4,5,6]:
            f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            # print(f_data['data'].shape)
            
            total_rows += f_data['data'].shape[1]  # 假设 'data' 数据集在轴1上的大小相同
            # print(total_rows)
            # 预分配数组空间

    # train_data_left= np.empty((total_rows, 1,36, 60))
    train_label= np.empty((total_rows, 2))  # 适应您的数据形状
    train_headpose= np.empty((total_rows, 1, 448, 448))
    # train_data_right = np.empty((total_rows,1,36, 60))  # 适应您的数据形状

    index = 0

    # 填充数据
    for ii in range(train_start,train_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)

            # f_data_left = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/two_branch_left_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            # f_data_right = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/two_branch_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

            f_label_head_pose = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/448_448_face_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    

            # f_data_left_temp = ((f_data_left['data'].value).T).reshape(-1, 1, 36, 60)
            # f_data_right_temp = ((f_data_right['data'].value).T).reshape(-1, 1, 36, 60)
            f_label_head_pose_temp = (( f_label_head_pose['data'].value).T).reshape(-1, 1, 448, 448)
            f_label_temp = ((f_label['data'].value)).T



            rows = f_label_temp.shape[0] 
            # print( rows)
            # train_data[index:index + rows, :, :, :] = train_data_temp
            # train_label[index:index + rows, :] = train_label_temp
            # train_voxel[index:index + rows, :, :, :] = train_voxel_temp


            # train_data_left[index:index + rows, :, :, :] =  f_data_left_temp 
            # train_data_right[index:index + rows, :, :, :] =  f_data_right_temp 
            train_label[index:index + rows, :,] = f_label_temp 
            train_headpose[index:index + rows, :] =  f_label_head_pose_temp 
            

            index += rows
    print(train_headpose.shape)
    # print(train_label.shape)
    # print(train_voxel.shape)


    trainDataset = Data.TensorDataset((torch.from_numpy(train_headpose).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_label).type(torch.FloatTensor)))


    n_train = len(trainDataset)
    
    trainloader  = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            # pin_memory=True
        )

###############################################################################################################################################

    total_rows = 0

    for ii in range(test_start,test_end):
        for jj in [1,2,3,4,5,6]:
            f_data = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            
            
            total_rows += f_data['data'].shape[1]  # 假设 'data' 数据集在轴1上的大小相同
            # print(total_rows)
            # 预分配数组空间

    # test_data_left= np.empty((total_rows, 1,36, 60))
    test_label= np.empty((total_rows, 2))  # 适应您的数据形状
    test_headpose= np.empty((total_rows, 1,448, 448))
    # test_data_right = np.empty((total_rows,1,36, 60))  # 适应您的数据形状

    index = 0

    # 填充数据
    for ii in range(test_start,test_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)

            f_label_head_pose = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/448_448_face_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label = h5py.File(os.path.join('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/data_for_gaze_network/no_scale_combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
    

            # f_data_left_temp = ((f_data_left['data'].value).T).reshape(-1, 1, 36, 60)
            # f_data_right_temp = ((f_data_right['data'].value).T).reshape(-1, 1, 36, 60)
            f_label_head_pose_temp = (( f_label_head_pose['data'].value).T).reshape(-1, 1, 448, 448)
            f_label_temp = ((f_label['data'].value)).T



            rows = f_label_temp.shape[0] 
            # print( rows)
            # test_data[index:index + rows, :, :, :] = test_data_temp
            # test_label[index:index + rows, :] = test_label_temp
            # test_voxel[index:index + rows, :, :, :] = test_voxel_temp


         
            test_label[index:index + rows, : ] = f_label_temp 
            test_headpose[index:index + rows, :] =  f_label_head_pose_temp 
            




            index += rows
    print(test_headpose.shape)
    # print(test_label.shape)
    # print(test_voxel.shape)


    testDataset = Data.TensorDataset( (torch.from_numpy(test_headpose).type(torch.FloatTensor)/ 255),(torch.from_numpy(test_label).type(torch.FloatTensor)))


    n_test = len(testDataset)
    
    testloader  = torch.utils.data.DataLoader(
            testDataset,
            batch_size=512,
            shuffle=False,
            num_workers=8,
            # pin_memory=True
        )
################################### Configure data loader#################################################


    if args.network == "gaitcnn":
        model =  gaitcnn.Net(1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
    elif args.network == "full_face":
        model =  full_face.model().to(device)
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

    with open(os.path.join("/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/", f"448_no_scale_full_face_first_left"), 'w') as outfile:
        for epoch in tqdm(range(args.epoch)):
            
            for i, data in enumerate(trainloader):
                model.train()
                x, labels= data
                x, labels = x.to(device), labels.to(device)
            
                pixel_preds =  model(x)

                loss= torch.nn.L1Loss()(pixel_preds,labels)          
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

                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    # print(labels.cpu().numpy())
                    label = model(inputs)
                    # print(label.cpu().numpy()[0])
                    # 取得分最高的那个类
                    Testloss += torch.nn.L1Loss()(label, labels)
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

                # resttime = (timeend - timebegin)/cur * (total-cur)/3600
                log = f"[{epoch}/{args.epoch}]: loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()   
                outfile.flush() 

            if epoch % 10== 0:
               torch.save(model.state_dict(), os.path.join('/home/sduu2/userspace/zgr/remote_eyetracking/python_code/Tpami_final_result/checkpoints/', f"448_Iter_{epoch}_no_scale_full_face_first_left.pt"))

