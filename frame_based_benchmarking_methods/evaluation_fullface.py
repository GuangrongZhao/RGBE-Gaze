import h5py
import numpy as np
import torch
import torch.utils.data as Data
import os
import sys
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from model import Full_face

def pitchyaw_to_vector(pitchyaws):
    """Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors."""
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles."""
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--data_path", default="G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_frame_method_eva/", help="Path to data files")
    parser.add_argument("--model_read_path", default="G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/checkpoints/", help="Path to pre-trained model checkpoint")
    parser.add_argument("--result_save_path", default="G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_frame_method_eva/ui_fullface/", help="Path to save results")
    args = parser.parse_args()

    cudnn.benchmark = True
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    print(device)

    userlist = [u for u in range(34, 67)]
    model = Full_face.model().to(device)
    state_dict = torch.load(os.path.join(args.model_read_path, 'fullface.pt'), map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)

    for user in userlist:


        total_rows = 0
        for ii in range(user, user + 1):
            for jj in [1, 2, 3, 4, 5, 6]:
                f_data = h5py.File(os.path.join(args.data_path, f"combine_pitch_yaw_user_{ii}_exp{jj}.h5"), 'r')
                total_rows += f_data['data'].shape[1]

        test_label = np.empty((total_rows, 2))
        test_headpose = np.empty((total_rows, 1, 448, 448))
        index = 0

        for ii in range(user, user + 1):
            for jj in [1, 2, 3, 4, 5, 6]:
                f_label_head_pose = h5py.File(os.path.join(args.data_path, f"448_448_face_normalized_user_{ii}_exp{jj}.h5"), 'r')
                f_label = h5py.File(os.path.join(args.data_path, f"combine_pitch_yaw_user_{ii}_exp{jj}.h5"), 'r')

                f_label_head_pose_temp = ((f_label_head_pose['data'][()]).T).reshape(-1, 1, 448, 448)
                f_label_temp = ((f_label['data'][()])).T
                rows = f_label_temp.shape[0]

                test_label[index:index + rows, :] = f_label_temp
                test_headpose[index:index + rows, :] = f_label_head_pose_temp
                index += rows

        testDataset = Data.TensorDataset((torch.from_numpy(test_headpose).type(torch.FloatTensor) / 255), (torch.from_numpy(test_label).type(torch.FloatTensor)))

        testloader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=512,
            shuffle=False,
            num_workers=8,
        )

        result_file = os.path.join(args.result_save_path, f"user_{user}")
        with open(result_file, 'w') as outfile:
            with torch.no_grad():
                model.eval()
                Testloss = 0
                Testlosstotal = 0
                Testdistance = 0

                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    label = model(inputs)

                    Testloss += torch.nn.MSELoss(reduction='sum')(label, labels)

                    output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))
                    Testdistance += np.sum((output))
                    Testlosstotal += (output.squeeze().shape[0])

                log = f"loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()
