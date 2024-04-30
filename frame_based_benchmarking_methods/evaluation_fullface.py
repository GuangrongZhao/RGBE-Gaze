import h5py
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import os
import sys
import argparse
sys.path.append("..")
import torch.backends.cudnn as cudnn

from model import Full_face

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
    parser.add_argument("--network", default="Full_face", help="The type of network")

    args = parser.parse_args()

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    print(device)


    #################################### Configure data loader#################################################
    userlist = [u for u in range(34,67)]
    model = Full_face.model().to(device)
    state_dict = torch.load(
        'G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/checkpoints/fullface_Iter_20.pt',
        map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)



    for user in userlist:
        print(user)
        
        total_rows = 0

        for ii in range(user,user+1):
            for jj in [1,2,3,4,5,6]:
                f_data = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_frame_method_eva/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                total_rows += f_data['data'].shape[1]
        test_label= np.empty((total_rows, 2))
        test_headpose= np.empty((total_rows, 1,448, 448))

        index = 0

        # 填充数据
        for ii in range(user,user+1):
            print(ii)
            for jj in [1,2,3,4,5,6]:
                # print(jj)

                f_label_head_pose = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_frame_method_eva/448_448_face_normalized_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f
                f_label = h5py.File(os.path.join('G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_frame_method_eva/combine_pitch_yaw_user_'+str(ii)+'_exp'+str(jj)+'.h5'),'r')   #创建一个h5文件，文件指针是f

                f_label_head_pose_temp = (( f_label_head_pose['data'][()]).T).reshape(-1, 1, 448, 448)
                # print(f_label_head_pose_temp[0,0,0])
                f_label_temp = ((f_label['data'][()])).T
                rows = f_label_temp.shape[0]
                test_label[index:index + rows, : ] = f_label_temp 
                test_headpose[index:index + rows, :] =  f_label_head_pose_temp
                index += rows
        print(test_headpose.shape)


        testDataset = Data.TensorDataset( (torch.from_numpy(test_headpose).type(torch.FloatTensor)/ 255),(torch.from_numpy(test_label).type(torch.FloatTensor)))


        n_test = len(testDataset)
        
        testloader  = torch.utils.data.DataLoader(
                testDataset,
                batch_size=512,
                shuffle=False,
                num_workers=8,
                # pin_memory=True
            )
    #################################### Configure data loader#################################################

        
        with open(os.path.join("G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/ui_fullface/user_" + str(user)), 'w') as outfile:

                with torch.no_grad():

                    model.eval()
                    Testloss = 0
                    Testlosstotal = 0
                    Testdistance = 0

                    kkk = 1
                    for data in testloader:  # 循环每一个batch

                        inputs, labels  = data
                        inputs, labels= inputs.to(device), labels.to(device)
                        label = model(inputs )

                        Testloss += torch.nn.MSELoss(reduction='sum')(label, labels)

                        output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))

                        Testdistance += np.sum((output))

                        Testlosstotal += (output.squeeze().shape[0])
                        kkk = kkk+1

                    log = f"loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush() 

