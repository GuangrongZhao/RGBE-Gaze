import h5py
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append("..")

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
        a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
        b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
        ab = np.sum(np.multiply(a, b), axis=1)
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)

        a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
        b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

        similarity = np.divide(ab, np.multiply(a_norm, b_norm))
        return np.arccos(similarity) * 180.0 / np.pi


    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=80, type=int, help="The number of epochs")
    parser.add_argument("--network", default="DilatedNet_voxel_grid", help="The type of network")
    parser.add_argument("--data_path", default="G:/remote_apperance_gaze_dataset/processed_data/data_network_training_for_event_method_eva/",
                        help="Path to the dataset")
    parser.add_argument("--result_save_path",
                        default="G:/remote_apperance_gaze_dataset/processed_data/pre_trained_models_for_event_method_eva/checkpoints",
                        help="Path to save the model checkpoints")

    args = parser.parse_args()

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
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

    total_rows = 0

    for ii in range(train_start, train_end):
        for jj in [1, 2, 3, 4, 5, 6]:
            f_data = h5py.File(os.path.join(args.data_path,
                                            f'R_pitch_yaw_user_{ii}_exp{jj}.h5'),'r')
            total_rows += f_data['data'].shape[1]

    train_data = np.empty((total_rows, 1, 64, 96))
    train_voxel = np.empty((total_rows, 5, 64, 96))
    train_label = np.empty((total_rows, 2))
    train_headpose = np.empty((total_rows, 2))

    index = 0

    for ii in range(train_start, train_end):
        print(ii)
        for jj in [1, 2, 3, 4, 5, 6]:
            f_data = h5py.File(os.path.join(args.data_path, f'64_96_right_eye_normalized_user_{ii}_exp{jj}.h5'), 'r')
            f_voxel = h5py.File(
                os.path.join(args.data_path, f'64_96_voxel_grid_right_eye_normalized_user_{ii}_exp{jj}.h5'), 'r')
            f_label = h5py.File(os.path.join(args.data_path, f'R_pitch_yaw_user_{ii}_exp{jj}.h5'), 'r')
            f_headpose = h5py.File(os.path.join(args.data_path, f'headpose_R_pitch_yaw_user_{ii}_exp{jj}.h5'), 'r')

            train_data_temp = ((f_data['data'][()]).T).reshape(-1, 1, 64, 96)
            train_voxel_temp = ((f_voxel['data'][()])).reshape(-1, 5, 64, 96)
            train_label_temp = ((f_label['data'][()]).T)
            train_headpose_temp = ((f_headpose['data'][()]).T)

            rows = train_data_temp.shape[0]
            train_data[index:index + rows, :, :, :] = train_data_temp
            train_label[index:index + rows, :] = train_label_temp
            train_voxel[index:index + rows, :, :, :] = train_voxel_temp
            train_headpose[index:index + rows, :] = train_headpose_temp

            index += rows
    print(train_data.shape)
    print(train_label.shape)
    print(train_voxel.shape)

    trainDataset = Data.TensorDataset(
        torch.from_numpy(train_data).type(torch.FloatTensor) / 255,
        torch.from_numpy(train_voxel).type(torch.FloatTensor),
        torch.from_numpy(train_label).type(torch.FloatTensor),
        torch.from_numpy(train_headpose).type(torch.FloatTensor)
    )

    n_train = len(trainDataset)

    trainloader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )

    total_rows = 0

    for ii in range(test_start, test_end):
        for jj in [1, 2, 3, 4, 5, 6]:
            f_data = h5py.File(os.path.join(args.data_path,
                                            f'R_pitch_yaw_user_{ii}_exp{jj}.h5'),
                               'r')
            total_rows += f_data['data'].shape[1]

    test_data = np.empty((total_rows, 1, 64, 96))
    test_voxel = np.empty((total_rows, 5, 64, 96))
    test_label = np.empty((total_rows, 2))
    test_headpose = np.empty((total_rows, 2))

    index = 0

    for ii in range(test_start, test_end):
        print(ii)
        for jj in [1, 2, 3, 4, 5, 6]:
            f_data = h5py.File(os.path.join(args.data_path, f'64_96_right_eye_normalized_user_{ii}_exp{jj}.h5'), 'r')
            f_voxel = h5py.File(
                os.path.join(args.data_path, f'64_96_voxel_grid_right_eye_normalized_user_{ii}_exp{jj}.h5'), 'r')
            f_label = h5py.File(os.path.join(args.data_path, f'R_pitch_yaw_user_{ii}_exp{jj}.h5'), 'r')
            f_headpose = h5py.File(os.path.join(args.data_path, f'headpose_R_pitch_yaw_user_{ii}_exp{jj}.h5'), 'r')

            test_data_temp = ((f_data['data'][()]).T).reshape(-1, 1, 64, 96)
            test_voxel_temp = ((f_voxel['data'][()])).reshape(-1, 5, 64, 96)
            test_label_temp = ((f_label['data'][()]).T)
            test_headpose_temp = ((f_headpose['data'][()]).T)

            rows = test_data_temp.shape[0]
            test_data[index:index + rows, :, :, :] = test_data_temp
            test_label[index:index + rows, :] = test_label_temp
            test_voxel[index:index + rows, :, :, :] = test_voxel_temp
            test_headpose[index:index + rows, :] = test_headpose_temp

            index += rows
    print(test_data.shape)
    print(test_label.shape)
    print(test_voxel.shape)

    testDataset = Data.TensorDataset(
        torch.from_numpy(test_data).type(torch.FloatTensor) / 255,
        torch.from_numpy(test_voxel).type(torch.FloatTensor),
        torch.from_numpy(test_label).type(torch.FloatTensor),
        torch.from_numpy(test_headpose).type(torch.FloatTensor)
    )

    n_test = len(testDataset)

    testloader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
    )

    #################################### Configure data loader#################################################

    model = DilatedNet_voxel_grid.DilatedNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    print(model)

    length = len(trainloader)

    with open(os.path.join(args.result_save_path, f"voxel_grid_R_eye"), 'w') as outfile:
        for epoch in tqdm(range(args.epoch)):
            for i, data in enumerate(trainloader):
                model.train()
                x, voxel, labels, head = data
                x, voxel, labels, head = x.to(device), voxel.to(device), labels.to(device), head.to(device)

                pixel_preds = model(x, voxel, head)

                loss = torch.nn.MSELoss(reduction='mean')(pixel_preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss)

            with torch.no_grad():
                model.eval()
                Testloss = 0
                Testlosstotal = 0
                Testdistance = 0

                for data in testloader:
                    inputs, voxel, labels, head = data
                    inputs, voxel, labels, head = inputs.to(device), voxel.to(device), labels.to(device), head.to(
                        device)

                    label = model(inputs, voxel, head)

                    Testloss += torch.nn.MSELoss(reduction='mean')(label, labels)

                    output = angular_error((label.cpu().detach().numpy()), (labels.cpu().numpy()))

                    Testdistance += np.mean(output)

                    Testlosstotal += 1

                log = f"[{epoch}/{args.epoch}]: loss:{Testloss / Testlosstotal} distance:{Testdistance / Testlosstotal}"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.result_save_path, f"voxel_grid_R_eye_Iter_{epoch}.pt"))
