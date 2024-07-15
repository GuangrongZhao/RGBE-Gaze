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

from model import Minist
from model import GazeNet

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
        a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
        b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
        ab = np.sum(np.multiply(a, b), axis=1)
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)
        a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
        b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)
        similarity = np.divide(ab, np.multiply(a_norm, b_norm))
        return np.arccos(similarity) * 180.0 / np.pi

    cudnn.benchmark = True
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--network", default="Minist", help="The type of network")
    parser.add_argument("--data_path", default="G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_frame_method_eva/", help="Path to the dataset")
    parser.add_argument("--model_read_path", default="G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/checkpoints/", help="Path to read pre-trained models")
    parser.add_argument("--result_save_path", default="G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/ui_minist/", help="Path to save results")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    print(device)



    model = Minist.model().to(device)
    state_dict = torch.load(os.path.join(args.model_read_path, 'minist_L_eye.pt'), map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    userlist = [u for u in range(34,67)]

    model.load_state_dict(state_dict)
    for user in userlist:
        print(user)

        test_data = []
        test_label = []
        flag = 1

        for ii in range(user, user + 1):
            for jj in [1, 2, 3, 4, 5, 6]:

                f_data = h5py.File(os.path.join(
                    args.data_path, f"36_60_L_eye_normalized_user_{ii}_exp{jj}.h5"), 'r')
                f_label = h5py.File(os.path.join(
                    args.data_path, f"L_pitch_yaw_user_{ii}_exp{jj}.h5"), 'r')
                f_head = h5py.File(os.path.join(
                    args.data_path, f"L_headpose_user_{ii}_exp{jj}.h5"), 'r')

                test_data_temp = ((f_data['data'][()]).T).reshape(-1, 1, 36, 60)
                test_label_temp = ((f_label['data'][()]).T)
                test_head_temp = ((f_head['data'][()]).T)
                if flag == 1:
                    test_data = test_data_temp
                    test_label = test_label_temp
                    test_head = test_head_temp
                    flag += 1
                else:
                    test_data = np.concatenate((test_data, test_data_temp), axis=0)
                    test_label = np.concatenate((test_label, test_label_temp), axis=0)
                    test_head = np.concatenate((test_head, test_head_temp), axis=0)

        print(test_data.shape)
        print(test_label.shape)
        print(test_head.shape)

        testDataset = Data.TensorDataset((torch.from_numpy(test_data).type(torch.FloatTensor) / 255),
                                         torch.from_numpy(test_label).type(torch.FloatTensor),
                                         torch.from_numpy(test_head).type(torch.FloatTensor))
        n_val = len(testDataset)
        testloader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=512,
            shuffle=False,
            num_workers=8
        )

        with open(os.path.join(args.result_save_path, f"user_{user}_L.txt"), 'w') as outfile:
            with torch.no_grad():
                model.eval()
                Testloss = 0
                Testlosstotal = 0
                Testdistance = 0

                for data in testloader:
                    inputs, labels, head = data
                    inputs, labels, head = inputs.to(device), labels.to(device), head.to(device)
                    label = model(inputs, head)
                    Testloss += torch.nn.MSELoss(reduction='sum')(label, labels)
                    output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))
                    Testdistance += np.sum((output))
                    Testlosstotal += (output.squeeze().shape[0])

                log = f"loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()
