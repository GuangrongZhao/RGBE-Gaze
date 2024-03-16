import os
from tqdm import tqdm
import numpy as np
import sys
from os.path import dirname, join
sys.path.append("..")
import random
import cv2
import scipy.io
import natsort
import concurrent.futures
import time
import h5py

# if __name__ == '__main__':
for user_name in range(1,67):
        for ex_time in range(5,7):
# def process_image(user_name, ex_time):
            print('user_name', user_name, 'ex_time', ex_time)
            # root_event_folder = r'D:/remote_dataset/user_%s/exp%s/prophesee' % (user_name, ex_time)
            # root_image_folder = r'D:/remote_dataset/user_%s/exp%s/convert2eventspace' % (user_name, ex_time)
            right_npz_file_list = []
            left_npz_file_list = []
            left_img_file_list = []
            right_img_file_list = []


            root_left_folder = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_for_eye_tracker/left_eye'%(user_name, ex_time)

            root_right_folder = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_for_eye_tracker/right_eye'%(user_name, ex_time)

            # left_jpgdatanames = [file for file in os.listdir( root_left_folder) if file.endswith('.jpg')]
            # left_jpgdatanames = natsort.natsorted(left_jpgdatanames )
            # # print(left_jpgdatanames)
            # right_jpgdatanames = [file for file in os.listdir( root_right_folder) if file.endswith('.jpg')]
            # right_jpgdatanames = natsort.natsorted(right_jpgdatanames )

            # left_npzdatanames = [file for file in os.listdir( root_left_folder) if file.startswith("accumulate_") and file.endswith('.npz')]
            left_npzdatanames = [file for file in os.listdir( root_left_folder) if file.startswith("") and file.endswith('.npz')]
            left_npzdatanames = natsort.natsorted(left_npzdatanames )
            print(left_npzdatanames)
            right_npzdatanames = [file for file in os.listdir( root_right_folder) if file.startswith("") and file.endswith('.npz')]
            right_npzdatanames = natsort.natsorted(right_npzdatanames )

            for kv in tqdm(range(0, len(right_npzdatanames))):
                # left_img_file_name = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation/user_%s/exp%s/voxel_grid_for_eye_tracker/left_eye/%s'% (user_name, ex_time, str( left_jpgdatanames [kv]))
                # print('load input face image: ', img_file_name)
                # print('load input face image: ', left_img_file_name)

                left_npz_file_name = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_for_eye_tracker/left_eye/%s' % (
                                user_name, ex_time, str(left_npzdatanames[kv]))


                # right_img_file_name = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation/user_%s/exp%s/voxel_grid_for_eye_tracker/right_eye/%s' % (
                # user_name, ex_time, str(right_jpgdatanames[kv]))
                # print('load input face image: ', img_file_name)
                # print('load input face image: ', right_img_file_name)

                # image = cv2.imread(right_img_file_name)
                right_npz_file_name = r'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_for_eye_tracker/right_eye/%s' % (
                    user_name, ex_time, str(right_npzdatanames[kv]))
                # print('load input face npz: ', right_npz_file_name)

                # left_image = cv2.imread(left_img_file_name, cv2.IMREAD_GRAYSCALE)
                # right_image = cv2.imread(right_img_file_name, cv2.IMREAD_GRAYSCALE)
                # left_image  = cv2.resize(left_image , (60, 36))
                #
                #
                # right_image = cv2.resize(right_image, (60, 36))
                # print(right_image)
                # cv2.imshow('Image', right_image)
                #
                # # 设置等待时间（单位：毫秒）
                # wait_time = 5000  # 5秒
                #
                # # 等待按键输入或超时
                # key = cv2.waitKey(wait_time)



                new_shape = (96, 64)
                left_events_bin = np.load(left_npz_file_name)
                left_events_bin_before =left_events_bin['event_bins'].astype(np.float32)


                left_events_bin_after = np.zeros((left_events_bin_before.shape[0], new_shape[1], new_shape[0]))




                # cv2.imshow('Image with Point22ss', left_events_bin_before[0, :, :] * 255)
                # cv2.waitKey(100)


                # print(left_events_bin_before.shape)
                for iii in range(5):
                    # 调整大小
                    left_events_bin_after[iii, :, :] = cv2.resize(left_events_bin_before[iii, :, :], new_shape,
                                                         interpolation=cv2.INTER_LINEAR)
                # print(left_events_bin_after.shape)



                #
                # voxel_grid_show =  left_events_bin_after[0, :, :] * 255
                # # 绘制点
                # # 显示图像
                # cv2.imshow('Image with Pointss', voxel_grid_show)
                # cv2.waitKey(100)






                right_events_bin = np.load(right_npz_file_name)
                right_events_bin_before = right_events_bin['event_bins'].astype(np.float32)

                right_events_bin_after = np.zeros((right_events_bin_before.shape[0], new_shape[1], new_shape[0]))

                # print(right_events_bin_before.shape)

                for iiii in range(5):
                    # 调整大小
                    right_events_bin_after[iiii, :, :] = cv2.resize(right_events_bin_before[iiii, :, :], new_shape,
                                                             interpolation=cv2.INTER_LINEAR)
                # print(right_events_bin_after.shape)





                # print(right_events_bin.shape)
                right_npz_file_list.append(right_events_bin_after)
                left_npz_file_list.append(right_events_bin_after)
                # right_img_file_list.append(right_image)
                #
                # # print(right_img_file_list[0])
                # left_img_file_list.append(left_image)

            # print((np.array(right_npz_file_list)).shape)
            # print((np.array(left_npz_file_list)).shape)
            # print((np.array(right_img_file_list)).shape)
            # print((np.array(left_img_file_list)).shape)
            # 打开现有的HDF5文件以供读取



            # np.savez_compressed('C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_voxel_grid/60_100_voxel_grid_left_eye_normalized_user_%s_exp%s.npz' %(user_name, ex_time), event_bins=np.array(left_npz_file_list))
            #
            # np.savez_compressed(
            #     'C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_voxel_grid/60_100_voxel_grid_right_eye_normalized_user_%s_exp%s.npz' % (
            #     user_name, ex_time), event_bins=np.array(right_npz_file_list))

            #


            left_npz_hdf5_file = h5py.File('C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_voxel_grid_smooth/64_96_voxel_grid_left_eye_normalized_user_%s_exp%s.h5' %(user_name, ex_time), "w")
            left_npz_hdf5_file.create_dataset("data", data=(np.array(left_npz_file_list)))


            # # left_image_hdf5_file = h5py.File('C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_gaze_network_voxel_grid/downsample_left_eye_normalized_user_%s_exp%s.h5' %(user_name, ex_time), "w")
            # # left_image_hdf5_file.create_dataset("data", data=(np.array(left_img_file_list)))




            right_npz_hdf5_file = h5py.File(
                'C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_voxel_grid_smooth/64_96_voxel_grid_right_eye_normalized_user_%s_exp%s.h5' % (
                user_name, ex_time), "w")
            right_npz_hdf5_file.create_dataset("data", data=(np.array(right_npz_file_list)))



            # print((np.array(right_npz_file_list).shape))
            # right_image_hdf5_file = h5py.File(
            #     'C:/Users/Aiot_Server/Desktop/remote_dataset/data_for_gaze_network_voxel_grid/downsample_right_eye_normalized_user_%s_exp%s.h5' % (user_name, ex_time),
            #     "w")
            # right_image_hdf5_file.create_dataset("data", data=(np.array(right_img_file_list)))

            # print(np.array(right_img_file_list))


                # print(right_events_bin)
                # right_npz_file_list.append(np.load(right_npz_file_name))
                #
                #
                # print((np.array(right_npz_file_list)).shape)

# def process_image_wrapper(args):
#     user_name, ex_time = args
#     process_image(user_name, ex_time)
#
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#         args_list = [(user_name, ex_time) for user_name in range(1, 2) for ex_time in range(1, 2)]
#         executor.map(process_image_wrapper, args_list)

