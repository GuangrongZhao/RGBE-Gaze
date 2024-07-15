# Convert event stream to voxelgrid representation

import os
from tqdm import tqdm
import numpy as npc
import sys
from os.path import dirname, join
sys.path.append("../..")
import random
import cv2
import scipy.io
import natsort
import concurrent.futures
import time
# import torch
import numpy as np

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1


    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    valid_indices &= tis >= 0

    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width+ tis[valid_indices] * width * height, vals_left[valid_indices])


    valid_indices = (tis + 1) < num_bins
    valid_indices &= tis >= 0
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width+ (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])


    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid




def find_first_greater_index(seq, target):
    left, right = 0, len(seq) - 1
    result = -1  # 初始化结果

    while left <= right:
        mid = (left + right) // 2
        if seq[mid] > target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result




base_path = 'G:/remote_apperance_gaze_dataset/'

for user_name in range(1, 67):
    for ex_time in range(1, 7):
        print('user_name', user_name, 'ex_time', ex_time)

        root_event_folder = f'{base_path}raw_data/user_{user_name}/exp{ex_time}/prophesee'
        root_image_folder = f'{base_path}raw_data/user_{user_name}/exp{ex_time}/convert2eventspace'
        root_output_folder = f'{base_path}processed_data/random_data_for_event_method_eva/voxel_grid_representation_2.5k/user_{user_name}/exp{ex_time}/voxel_grid_saved'
        timestamp_output_folder = f'{root_output_folder}/inter_timestamp.txt'
        random_sequence_read_folder = f'{base_path}event_eva_random_index/user_{user_name}_exp_{ex_time}.txt'
        image_align_event_time = f'{base_path}processed_data/random_data_for_event_method_eva/frame_align_event_timestamp/user_{user_name}_exp_{ex_time}.txt'

        left_rv_gazenorm_list = scipy.io.loadmat(
            f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/left_eye/left_rv_gazenorm_list.mat')['left_rv_gazenorm_list']

        left_scale_vector_list = scipy.io.loadmat(
            f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/left_eye/left_scale_vector_list.mat')['left_scale_vector_list']

        right_rv_gazenorm_list = scipy.io.loadmat(
            f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/right_eye/right_rv_gazenorm_list.mat')['right_rv_gazenorm_list']

        right_scale_vector_list = scipy.io.loadmat(
            f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/right_eye/right_scale_vector_list.mat')['right_scale_vector_list']

        with open(image_align_event_time) as f:
            timestamp = [float(i) for i in f.readlines()]

        with open(random_sequence_read_folder) as f:
            random_sequence = [float(i) for i in f.readlines()]
        random_sequence = np.array(random_sequence)

        events = np.load(root_event_folder + "/event_merge.npz", allow_pickle=True)
        (x, y, t, p) = (
            events["x"].astype(np.float64).reshape((-1,)),
            events["y"].astype(np.float64).reshape((-1,)),
            events["t"].astype(np.float64).reshape((-1,)),
            events["p"].astype(np.float32).reshape((-1,)) * 2 - 1,
        )

        datanames = [file for file in os.listdir(root_image_folder) if file.endswith('.jpg')]
        datanames = natsort.natsorted(datanames)

        focal_norm = 1800  # focal length of normalized camera
        roiSize = (1280, 720)
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])

        camera_matrix = np.array([[1.712706065476114e+03, 0, 0],
                                  [0, 1713.40524379986, 0],
                                  [6.560971442117219e+02, 3.451634978903322e+02, 1]])
        cam = camera_matrix.T

        xx = 640 - 50  # Center x-coordinate minus half the width
        yy = 360 - 30  # Center y-coordinate minus half the height
        width = 100
        height = 60
        save_index = 0

        shape = (5, 60, 100)
        build_voxel_grid = np.zeros(shape)

        with open(timestamp_output_folder, 'w') as file:
            for j in tqdm(range(0, len(random_sequence))):
                sequence_index = int(float(random_sequence[j]))

                left_eye_img_file_name = f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/left_eye/{datanames[sequence_index]}'
                left_eye_image = cv2.imread(left_eye_img_file_name)  # Read raw image
                left_eye_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)

                right_eye_img_file_name = f'{base_path}processed_data/random_data_for_event_method_eva/frame/user_{user_name}/exp{ex_time}/convert_frame_normalized/right_eye/{datanames[sequence_index]}'
                right_eye_image = cv2.imread(right_eye_img_file_name)  # Read raw right image
                right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)

                left_W = np.dot(np.dot(cam_norm, left_scale_vector_list[j]), np.dot(left_rv_gazenorm_list[j], np.linalg.inv(cam)))
                right_W = np.dot(np.dot(cam_norm, right_scale_vector_list[j]), np.dot(right_rv_gazenorm_list[j], np.linalg.inv(cam)))

                for i in range(0, 50):  # Insert 50 voxel grids between every two frames
                    t_start_index_fast = find_first_greater_index(t, timestamp[sequence_index])
                    t_end_index_fast = find_first_greater_index(t, timestamp[sequence_index] + ((i + 1) * 400))
                    t_for_voxel_grid_timestamp = timestamp[sequence_index] + ((i + 1) * 400)

                    x_before_warp = x[t_start_index_fast:t_end_index_fast + 1]
                    y_before_warp = y[t_start_index_fast:t_end_index_fast + 1]
                    t_before_warp = t[t_start_index_fast:t_end_index_fast + 1]
                    p_before_warp = p[t_start_index_fast:t_end_index_fast + 1]

                    x_y_before_warp = np.vstack((x_before_warp, y_before_warp)).T
                    x_y_before_warp = x_y_before_warp.reshape(x_y_before_warp.shape[0], 1, 2)

                    if len(x_y_before_warp) == 0:
                        x_y_before_warp = np.zeros((2, 1, 2))
                        t_before_warp = np.zeros((2,))
                        p_before_warp = np.zeros((2,))

                    x_y_warped_left = cv2.perspectiveTransform(x_y_before_warp, left_W)
                    x_y_warped_right = cv2.perspectiveTransform(x_y_before_warp, right_W)

                    x_y_warped_left = x_y_warped_left.reshape(x_y_before_warp.shape[0], 2).astype(int)
                    x_y_warped_right = x_y_warped_right.reshape(x_y_before_warp.shape[0], 2).astype(int)

                    left_crop_index = (x_y_warped_left[:, 0] >= 590) & (x_y_warped_left[:, 0] < 690) & (x_y_warped_left[:, 1] >= 330) & (x_y_warped_left[:, 1] < 390)
                    right_crop_index = (x_y_warped_right[:, 0] >= 590) & (x_y_warped_right[:, 0] < 690) & (x_y_warped_right[:, 1] >= 330) & (x_y_warped_right[:, 1] < 390)

                    x_y_warped_left_crop = x_y_warped_left[left_crop_index]
                    x_y_warped_right_crop = x_y_warped_right[right_crop_index]

                    x_y_warped_left_crop -= np.array([590, 330])  # Use the upper left corner of the crop box as the origin
                    x_y_warped_right_crop -= np.array([590, 330])

                    t_warped_crop_left = t_before_warp[left_crop_index]
                    p_warped_crop_left = p_before_warp[left_crop_index]

                    t_warped_crop_right = t_before_warp[right_crop_index]
                    p_warped_crop_right = p_before_warp[right_crop_index]

                    event_input_left = np.column_stack((t_warped_crop_left, x_y_warped_left_crop, p_warped_crop_left))
                    event_input_right = np.column_stack((t_warped_crop_right, x_y_warped_right_crop, p_warped_crop_right))

                    if len(event_input_left) != 0:
                        event_output_left = events_to_voxel_grid(event_input_left, 5, 100, 60)
                    else:
                        event_output_left = build_voxel_grid
                        print("造bin", event_output_left.shape)

                    np.savez_compressed(f'{root_output_folder}/left_eye/{save_index:06}.npz', event_bins=event_output_left.astype(np.float32))
                    savepath = f'{root_output_folder}/left_eye/{save_index:06}.jpg'
                    cv2.imwrite(savepath, left_eye_image)

                    if len(event_input_right) != 0:
                        event_output_right = events_to_voxel_grid(event_input_right, 5, 100, 60)
                    else:
                        event_output_right = build_voxel_grid
                        print("造bin", event_output_right.shape)

                    np.savez_compressed(f'{root_output_folder}/right_eye/{save_index:06}.npz', event_bins=event_output_right.astype(np.float32))
                    savepath = f'{root_output_folder}/right_eye/{save_index:06}.jpg'
                    cv2.imwrite(savepath, right_eye_image)

                    file.write(str(t_for_voxel_grid_timestamp) + '\n')

                    save_index += 1



