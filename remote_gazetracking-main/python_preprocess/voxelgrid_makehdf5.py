import os
from tqdm import tqdm
import numpy as np
import sys
from os.path import dirname, join
sys.path.append("../..")
import random
import cv2
import scipy.io
import natsort
import concurrent.futures
import time
import h5py

import os
from tqdm import tqdm
import numpy as np
import sys
from os.path import dirname, join
sys.path.append("../..")
import random
import cv2
import scipy.io
import natsort
import concurrent.futures
import time
import h5py

datasetpath = 'G:/remote_apperance_gaze_dataset/'

# if __name__ == '__main__':
for user_name in range(1, 67):
    for ex_time in range(1, 7):
        # def process_image(user_name, ex_time):
        print('user_name', user_name, 'ex_time', ex_time)
        right_npz_file_list = []
        left_npz_file_list = []
        left_img_file_list = []
        right_img_file_list = []

        root_left_folder = os.path.join(datasetpath, 'processed_data/random_data_for_event_method_eva/voxel_grid_representation_2.5k/user_%s/exp%s/voxel_grid_for_eye_tracker/left_eye' % (user_name, ex_time))

        root_right_folder = os.path.join(datasetpath, 'processed_data/random_data_for_event_method_eva/voxel_grid_representation_2.5k/user_%s/exp%s/voxel_grid_for_eye_tracker/right_eye' % (user_name, ex_time))

        left_npzdatanames = [file for file in os.listdir(root_left_folder) if file.startswith("") and file.endswith('.npz')]
        left_npzdatanames = natsort.natsorted(left_npzdatanames)
        print(left_npzdatanames)
        right_npzdatanames = [file for file in os.listdir(root_right_folder) if file.startswith("") and file.endswith('.npz')]
        right_npzdatanames = natsort.natsorted(right_npzdatanames)

        for kv in tqdm(range(0, len(right_npzdatanames))):
            left_npz_file_name = os.path.join(datasetpath, 'processed_data/random_data_for_event_method_eva/voxel_grid_representation_2.5k/user_%s/exp%s/voxel_grid_for_eye_tracker/left_eye/%s' % (user_name, ex_time, str(left_npzdatanames[kv])))

            right_npz_file_name = os.path.join(datasetpath, 'processed_data/random_data_for_event_method_eva/voxel_grid_representation_2.5k/user_%s/exp%s/voxel_grid_for_eye_tracker/right_eye/%s' % (user_name, ex_time, str(right_npzdatanames[kv])))

            new_shape = (96, 64)
            left_events_bin = np.load(left_npz_file_name)
            left_events_bin_before = left_events_bin['event_bins'].astype(np.float32)

            left_events_bin_after = np.zeros((left_events_bin_before.shape[0], new_shape[1], new_shape[0]))

            for iii in range(5):
                # 调整大小
                left_events_bin_after[iii, :, :] = cv2.resize(left_events_bin_before[iii, :, :], new_shape,
                                                              interpolation=cv2.INTER_LINEAR)

            right_events_bin = np.load(right_npz_file_name)
            right_events_bin_before = right_events_bin['event_bins'].astype(np.float32)

            right_events_bin_after = np.zeros((right_events_bin_before.shape[0], new_shape[1], new_shape[0]))

            for iiii in range(5):
                # 调整大小
                right_events_bin_after[iiii, :, :] = cv2.resize(right_events_bin_before[iiii, :, :], new_shape,
                                                                interpolation=cv2.INTER_LINEAR)

            right_npz_file_list.append(right_events_bin_after)
            left_npz_file_list.append(left_events_bin_after)

        left_npz_hdf5_file = h5py.File(os.path.join(datasetpath, 'processed_data/data_network_training_for_event_method_eva/64_96_voxel_grid_left_eye_normalized_user_%s_exp%s.h5' % (user_name, ex_time)), "w")
        left_npz_hdf5_file.create_dataset("data", data=(np.array(left_npz_file_list)))

        right_npz_hdf5_file = h5py.File(os.path.join(datasetpath, 'processed_data/data_network_training_for_event_method_eva/64_96_voxel_grid_right_eye_normalized_user_%s_exp%s.h5' % (user_name, ex_time)), "w")
        right_npz_hdf5_file.create_dataset("data", data=(np.array(right_npz_file_list)))

