import h5py
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import os
import sys
import argparse
sys.path.append("..")
from model import RTGene
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

   
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--testuser", default="0", help="The GPU ID")
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=60, type=int, help="The number of epochs")
    parser.add_argument("--network", default="RTGene", help="The type of network")

    args = parser.parse_args()

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    print(device)


    #################################### Configure data loader#################################################


    train_start = 1
    train_end = 34

    test_start = 34
    test_end = 67

    total_rows = 0

    for ii in range(train_start,train_end):
        for jj in [1,2,3,4,5,6]:
            f_data = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

            total_rows += f_data['data'].shape[1]  # 假设 'data' 数据集在轴1上的大小相同
    train_data_left= np.empty((total_rows, 1,36, 60))
    train_label= np.empty((total_rows, 2))  # 适应您的数据形状
    train_headpose= np.empty((total_rows,2))  # 适应您的数据形状
    train_data_right = np.empty((total_rows,1,36, 60))  # 适应您的数据形状

    index = 0

    # 填充数据
    for ii in range(train_start,train_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)

            f_data_left = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/two_branch_left_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_data_right = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/two_branch_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label_head_pose = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/face_norm_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

            f_data_left_temp = ((f_data_left['data'][()]).T).reshape(-1, 1, 36, 60)
            f_data_right_temp = ((f_data_right['data'][()]).T).reshape(-1, 1, 36, 60)
            f_label_head_pose_temp = ((f_label_head_pose['data'][()])).T
            f_label_temp = ((f_label['data'][()])).T

            rows = f_data_left_temp.shape[0]
            train_data_left[index:index + rows, :, :, :] =  f_data_left_temp
            train_data_right[index:index + rows, :, :, :] =  f_data_right_temp
            train_label[index:index + rows, :,] = f_label_temp
            train_headpose[index:index + rows, :] =  f_label_head_pose_temp

            index += rows
    print(train_data_left.shape)

    trainDataset = Data.TensorDataset( (torch.from_numpy(train_data_left).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_data_right).type(torch.FloatTensor)/ 255),(torch.from_numpy(train_label).type(torch.FloatTensor)),(torch.from_numpy(train_headpose).type(torch.FloatTensor)))


    n_train = len(trainDataset)

    trainloader  = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=256,
            shuffle=True,
            num_workers=8,
            # pin_memory=True
        )

####################################### #########################################################################################################

    total_rows = 0

    for ii in range(test_start,test_end):
        for jj in [1,2,3,4,5,6]:
            f_data = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

            total_rows += f_data['data'].shape[1]

    test_data_left= np.empty((total_rows, 1,36, 60))
    test_label= np.empty((total_rows, 2))
    test_headpose= np.empty((total_rows,2))
    test_data_right = np.empty((total_rows,1,36, 60))

    index = 0

    # 填充数据
    for ii in range(test_start,test_end):
        print(ii)
        for jj in [1,2,3,4,5,6]:
            # print(jj)

            f_data_left = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/two_branch_left_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_data_right = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/two_branch_right_eye_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label_head_pose = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/face_norm_headpose_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
            f_label = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_for_frame_gaze_network_training/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

            f_data_left_temp = ((f_data_left['data'][()]).T).reshape(-1, 1, 36, 60)
            f_data_right_temp = ((f_data_right['data'][()]).T).reshape(-1, 1, 36, 60)
            f_label_head_pose_temp = ((f_label_head_pose['data'][()])).T
            f_label_temp = ((f_label['data'][()])).T

            rows = f_data_left_temp.shape[0]
            test_data_left[index:index + rows, :, :, :] =  f_data_left_temp
            test_data_right[index:index + rows, :, :, :] =  f_data_right_temp
            test_label[index:index + rows, : ] = f_label_temp
            test_headpose[index:index + rows, :] =  f_label_head_pose_temp
            index += rows
    print(test_data_left.shape)
    testDataset = Data.TensorDataset( (torch.from_numpy(test_data_left).type(torch.FloatTensor)/ 255),(torch.from_numpy(test_data_right).type(torch.FloatTensor)/ 255),(torch.from_numpy(test_label).type(torch.FloatTensor)),(torch.from_numpy(test_headpose).type(torch.FloatTensor)))
    n_test = len(testDataset)

    testloader  = torch.utils.data.DataLoader(
            testDataset,
            batch_size=512,
            shuffle=False,
            num_workers=8,
            # pin_memory=True
        )
#################################### Configure data loader#################################################


    if args.network == "RTGene":
        model = RTGene.model().to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr= 0.0001, betas=(0.9,0.95))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 70000, gamma=1)


    length = len(trainloader)

    with open(os.path.join("G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models/", f"rtgene"), 'w') as outfile:
        for epoch in tqdm(range(args.epoch)):

            for i, data in enumerate(trainloader):
                model.train()
                x_left, x_right,labels,headpose = data
                x_left, x_right,labels,headpose= x_left.to(device), x_right.to(device),labels.to(device),headpose.to(device)
                pixel_preds=  model( x_left, x_right,headpose)
                loss= torch.nn.MSELoss(reduction='mean')(pixel_preds,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():

                model.eval()
                Testloss = 0
                Testlosstotal = 0
                Testdistance = 0

                kkk = 1
                for data in testloader:

                    x_left, x_right,labels,headpose = data
                    x_left, x_right,labels,headpose= x_left.to(device), x_right.to(device),labels.to(device),headpose.to(device)
                    label =  model( x_left, x_right,headpose)
                    Testloss += torch.nn.MSELoss(reduction='mean')(label, labels)
                    output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))
                    Testdistance += np.mean(output)
                    Testlosstotal += 1
                    kkk = kkk+1

                log = f"[{epoch}/{args.epoch}]: loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()

            if epoch % 5== 0:
                 torch.save(model.state_dict(), os.path.join('G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models/checkpoints/', f"rtgene_Iter_{epoch}.pt"))

