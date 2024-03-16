import os
from tqdm import tqdm
import numpy as npc
import sys
from os.path import dirname, join
sys.path.append("..")
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

    # assert(events.shape[1] == 4)
    # assert(num_bins > 0)
    # assert(width > 0)
    # assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # print(voxel_grid.shape)
    # normalize the event timestamps so that they lie between 0 and num_bins
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
    # print(xs[valid_indices])
    # print(ys[valid_indices])
    # print(tis[valid_indices])
    # print(vals_left[valid_indices])
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width+ tis[valid_indices] * width * height, vals_left[valid_indices])


    valid_indices = (tis + 1) < num_bins
    valid_indices &= tis >= 0
    # print(valid_indices)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width+ (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])


    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))


    # non_zero_elements = voxel_grid[voxel_grid != 0]
    # non_zero_indices = np.where(non_zero_elements != 0)
    # # 打印结果
    # print(non_zero_indices)
    # print(non_zero_elements)
    # time.sleep(100)
    return voxel_grid
#

# def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    # DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    # assert(events.shape[1] == 4)
    # assert(num_bins > 0)
    # assert(width > 0)
    # assert(height > 0)

    # with torch.no_grad():
    #
    #     events_torch = torch.from_numpy(events)
    #     device = 'cpu'
    #     events_torch = events_torch.to(device)
    #
    #     # with DeviceTimer('Voxel grid voting'):
    #     voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()
    #
    #     # normalize the event timestamps so that they lie between 0 and num_bins
    #     last_stamp = events_torch[-1, 0]
    #     first_stamp = events_torch[0, 0]
    #     deltaT = last_stamp - first_stamp
    #
    #     if deltaT == 0:
    #         deltaT = 1.0
    #
    #     events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
    #     ts = events_torch[:, 0]
    #     xs = events_torch[:, 1].long()
    #     ys = events_torch[:, 2].long()
    #     pols = events_torch[:, 3].float()
    #     pols[pols == 0] = -1  # polarity should be +1 / -1
    #
    #     tis = torch.floor(ts)
    #     tis_long = tis.long()
    #     dts = ts - tis
    #     vals_left = pols * (1.0 - dts.float())
    #     vals_right = pols * dts.float()
    #
    #     valid_indices = tis < num_bins
    #     valid_indices &= tis >= 0
    #     voxel_grid.index_add_(dim=0,
    #                           index=xs[valid_indices] + ys[valid_indices]
    #                           * width + tis_long[valid_indices] * width * height,
    #                           source=vals_left[valid_indices])
    #
    #     valid_indices = (tis + 1) < num_bins
    #     valid_indices &= tis >= 0
    #
    #     voxel_grid.index_add_(dim=0,
    #                           index=xs[valid_indices] + ys[valid_indices] * width
    #                           + (tis_long[valid_indices] + 1) * width * height,
    #                           source=vals_right[valid_indices])
    #
    #     voxel_grid = voxel_grid.view(num_bins, height, width)
    #
    #     voxel_grid = voxel_grid.numpy()
    #
    #     # non_zero_elements = voxel_grid[voxel_grid != 0]
    #     # non_zero_indices = np.where(non_zero_elements != 0)
    #     # # 打印结果
    #     # print(non_zero_indices)
    #     # print(non_zero_elements)
    #     # time.sleep(100)
    # return voxel_grid




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




# if __name__ == '__main__':2
for user_name in range(1, 67):
        for ex_time in range(5, 7):
# def process_image(user_name, ex_time):
            print('user_name', user_name, 'ex_time', ex_time)
            root_event_folder = 'D:/remote_dataset/user_%s/exp%s/prophesee' % (user_name, ex_time)
            root_image_folder = 'D:/remote_dataset/user_%s/exp%s/convert2eventspace' % (user_name, ex_time)
            root_output_folder = 'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_saved' % (user_name, ex_time)
            timestamp_output_folder = 'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_representation_smooth/user_%s/exp%s/voxel_grid_saved/inter_timestamp.txt' % (user_name, ex_time)
            random_sequence_read_folder = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/random_numbers.txt' % (user_name, ex_time)

            image_align_event_time = 'D:remote_dataset/user_%s/exp%s/convert2eventspace/frame_align_event_timestamp.txt' % (user_name, ex_time)

            left_rv_gazenorm_list = scipy.io.loadmat(
                'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/left_rv_gazenorm_list.mat' % (
                user_name, ex_time))
            left_rv_gazenorm_list= left_rv_gazenorm_list['left_rv_gazenorm_list']
            # print(left_rv_gazenorm_list[0])

            left_scale_vector_list = scipy.io.loadmat(
                'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/left_scale_vector_list.mat' % (
                    user_name, ex_time))
            left_scale_vector_list = left_scale_vector_list['left_scale_vector_list']
            # print(left_scale_vector_list[0])

            right_rv_gazenorm_list = scipy.io.loadmat(
                'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/right_rv_gazenorm_list.mat' % (
                    user_name, ex_time))
            right_rv_gazenorm_list = right_rv_gazenorm_list['right_rv_gazenorm_list']
            # print(right_rv_gazenorm_list[0])

            right_scale_vector_list = scipy.io.loadmat(
                'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/right_scale_vector_list.mat' % (
                    user_name, ex_time))
            right_scale_vector_list = right_scale_vector_list['right_scale_vector_list']
            # print(right_scale_vector_list[0])


            with open(image_align_event_time) as f:
                timestamp = [float(i) for i in f.readlines()]
                # print(timestamp)

            with open(random_sequence_read_folder) as f:
                random_sequence = [float(i) for i in f.readlines()]
            random_sequence = np.array(random_sequence)
            # print(random_sequence)
            events = np.load(root_event_folder + "/event_merge.npz", allow_pickle=True)

            (x, y, t, p) = (
                events["x"].astype(np.float64).reshape((-1,)),
                events["y"].astype(np.float64).reshape((-1,)),
                events["t"].astype(np.float64).reshape((-1,)),
                events["p"].astype(np.float32).reshape((-1,)) * 2 - 1,
            )
            # # events = np.stack((t,x, y, polarity), axis=-1)  # 获得events

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

            xx = 640 - 50  # 中心横坐标减去宽度的一半
            yy = 360 - 30  # 中心纵坐标减去高度的一半
            width = 100
            height = 60
            save_index = 0

            shape = (5, 60, 100)
            build_voxel_grid = np.zeros(shape)

            output_path = f'C:/Users/Aiot_Server/Desktop/remote_dataset/voxel_grid_video/smooth_users{user_name}_video.mp4'
            # frame_rate = 60.0  # 帧速率（每秒多少帧）
            # frame_size = (1280, 720)  # 帧大小（宽度，高度）
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
            # out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)
            frame_list = []

            with open(timestamp_output_folder, 'w') as file:

             for j in tqdm(range(0,len(random_sequence))):
                # print(j)

                # print('user_name', user_name, 'ex_time', ex_time)
                sequence_index = int(random_sequence[j])
                # print(sequence_index )
                # print(timestamp[sequence_index])


                img_file_name = 'D:/remote_dataset/user_%s/exp%s/convert2eventspace/%s' % (
                user_name, ex_time, str(datanames[ sequence_index ]))
                # print('load input face image: ', img_file_name)
                image = cv2.imread(img_file_name)  # 读取raw image



                left_eye_img_file_name = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/%s' % (
                    user_name, ex_time, str(datanames[sequence_index]))
                # print('load input face image: ',  left_eye_img_file_name)
                left_eye_image = cv2.imread(left_eye_img_file_name)  # 读取raw image
                left_eye_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2GRAY)


                right_eye_img_file_name = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/%s' % (
                    user_name, ex_time, str(datanames[sequence_index]))
                # print('load input face right: ', right_eye_img_file_name)
                right_eye_image = cv2.imread(right_eye_img_file_name)  # 读取raw right
                right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)



                left_W  = np.dot(np.dot(cam_norm, left_scale_vector_list[j]), np.dot(left_rv_gazenorm_list[j], np.linalg.inv(cam)))

                right_W = np.dot(np.dot(cam_norm, right_scale_vector_list[j]),np.dot(right_rv_gazenorm_list[j], np.linalg.inv(cam)))

                # print(left_W)

                # print(right_W)



                # img_warped_left = cv2.warpPerspective(image, left_W, roiSize)  # image normalization
                # cv2.imshow("img", img_warped_left)
                # cv2.waitKey(5000)


                for i in range(0,19): # 每 两张插入 19个  voxel grid

                    # start_time = time.time()

                    # t_satrt_index = np.where(t > timestamp[sequence_index]+ i*1000)   #根据frame查找事件
                    # t_end_index = np.where(t > timestamp[sequence_index] + (i+1) * 1000)


                    # print(t_for_voxel_grid_timestamp)
                    # end_time = time.time()
                    #
                    # # 计算运行时间
                    # execution_time = end_time - start_time
                    #
                    # print("程序运行时间：", execution_time, "秒")


                    t_satrt_index_fast = find_first_greater_index(t, timestamp[sequence_index])
                    t_end_index_fast = find_first_greater_index(t, timestamp[sequence_index] + ((i + 1) * 1000))

                    # print(t_satrt_index)
                    # print(t_satrt_index_fast)

                    t_for_voxel_grid_timestamp = timestamp[sequence_index] + ((i + 1) * 1000)



                    # print('t(t_satrt)', t_satrt_index[0][0])
                    # print('t(t_end)', t_end_index[0][0]+1)

                    # t_satrt_index[0][0]
                    # t_end_index[0][0]

                    x_before_warp = x[( t_satrt_index_fast):( t_end_index_fast )+1]
                    y_before_warp = y[( t_satrt_index_fast):( t_end_index_fast )+1]

                    t_before_warp = t[( t_satrt_index_fast):( t_end_index_fast ) + 1]
                    p_before_warp = p[( t_satrt_index_fast):( t_end_index_fast ) + 1]
                    # t_mean_for_voxel_grid_record = np.mean(t[(t_satrt_index[0][0]):(t_end_index[0][0])+1])



                    x_y_before_warp = np.vstack(( x_before_warp , y_before_warp))
                    x_y_before_warp = x_y_before_warp.T
                    x_y_before_warp = x_y_before_warp.reshape(x_y_before_warp.shape[0], 1, 2)
                    # print('t(t_end)', x_y_before_warp.shape)
                    # print('t(t_end)', t_before_warp.shape)
                    # print('t(t_end)', p_before_warp.shape)
                    if len(x_y_before_warp) == 0:
                        x_y_before_warp = np.zeros((2, 1, 2))
                        t_before_warp = np.zeros((2, ))
                        p_before_warp = np.zeros((2, ))
                        # print('t(t_end)', x_y_before_warp.shape)
                        # print('t(t_end)', t_before_warp.shape)
                        # print('t(t_end)', p_before_warp.shape)

                    x_y_warped_left = cv2.perspectiveTransform(x_y_before_warp,left_W)
                    x_y_warped_right = cv2.perspectiveTransform(x_y_before_warp,right_W)

                    x_y_warped_left = x_y_warped_left.reshape(x_y_before_warp.shape[0], 2).astype(int)
                    x_y_warped_right = x_y_warped_right.reshape(x_y_before_warp.shape[0], 2).astype(int)
                    # print(x_y_warped_right.shape)

                    left_crop_index = (x_y_warped_left[:, 0] >= 590 ) & (x_y_warped_left[:, 0] < 690) & (x_y_warped_left[:, 1] >= 330) & (x_y_warped_left[:, 1] < 390)
                    right_crop_index = (x_y_warped_right[:, 0] >= 590) & (x_y_warped_right[:, 0] < 690) & (
                                x_y_warped_right[:, 1] >= 330) & (x_y_warped_right[:, 1] < 390)


                    x_y_warped_left_crop =  x_y_warped_left[left_crop_index]
                    # print(x_y_warped_left_crop)
                    x_y_warped_right_crop = x_y_warped_right[right_crop_index]



                    # color = (0, 0, 255)  # 红色
                    # img_warped_left = cv2.warpPerspective(image, left_W, roiSize)  # image normalization
                    # # 绘制点
                    # for point in x_y_warped_left_crop:
                    #     cv2.circle(img_warped_left, point, 2, color, -1)  # 半径为5，-1 表示填充整个圆
                    #
                    # # 显示图像
                    # cv2.imshow('Image with Points', img_warped_left)
                    # cv2.waitKey(100)

                    # frame_list.append(img_warped_left)

                    # 将图像帧写入视频



                    x_y_warped_left_crop -= np.array([590, 330]) # 以crop 方格左上角为起点
                    x_y_warped_right_crop -= np.array([590, 330])



                    # print('x_y_warped_left_crop', x_y_warped_left_crop)
                    # print(left_crop_index)
                    t_warped_crop_left = t_before_warp[left_crop_index]
                    # print(t_warped_crop_left)

                    # print(t_warped_crop_left)
                    p_warped_crop_left = p_before_warp[left_crop_index]

                    t_warped_crop_right = t_before_warp[right_crop_index]

                    p_warped_crop_right = p_before_warp[right_crop_index]

                    event_input_left = np.column_stack((t_warped_crop_left, x_y_warped_left_crop, p_warped_crop_left))
                    event_input_right = np.column_stack((t_warped_crop_right, x_y_warped_right_crop, p_warped_crop_right))
                    # print(event_input_left.shape)



                    if len(event_input_left) != 0:
                        event_output_left = events_to_voxel_grid(event_input_left, 5, 100, 60)
                        # print(event_output_left.shape)

                    else:
                        # 生成一个全零矩阵

                        event_output_left = build_voxel_grid
                        # 将每个 (720, 1280) 矩阵的第一个元素设置为 1
                        # for i in range(shape[0]):
                        #     matrix[i, 0, 0] = 1
                        # 输出结果
                        print("造bin",event_output_left.shape)

                    np.savez_compressed(root_output_folder + '/left_eye/{:06}.npz'.format(save_index
                       ), event_bins=event_output_left.astype(np.float32))

                    # print(right_eye_image.shape)


                    #
                    savepath = (root_output_folder + '/left_eye/{:06}.jpg'.format(save_index))
                    # 保存灰度图像
                    cv2.imwrite( savepath, left_eye_image)

                    if len(event_input_right) != 0:
                        event_output_right = events_to_voxel_grid(event_input_right, 5, 100, 60)
                        # print(event_output_right.shape)

                    else:
                        # 生成一个全零矩阵

                        event_output_right =  build_voxel_grid
                        # 将每个 (720, 1280) 矩阵的第一个元素设置为 1
                        # for i in range(shape[0]):
                        #     matrix[i, 0, 0] = 1
                        # 输出结果
                        print("造bin",event_output_right.shape)



                    # color = (0, 0, 255)  # 红色
                    # voxel_grid_show = event_output_left[0,:,:]*255
                    # # 绘制点
                    #
                    #
                    # # 显示图像
                    # cv2.imshow('Image with Pointss', voxel_grid_show)
                    # cv2.waitKey(100)




                    np.savez_compressed(root_output_folder + '/right_eye/{:06}.npz'.format(save_index
                       ), event_bins=event_output_right.astype(np.float32))

                    # print(right_eye_image.shape)
                    #


                    savepath = (root_output_folder + '/right_eye/{:06}.jpg'.format(save_index))
                    # 保存灰度图像
                    cv2.imwrite( savepath, right_eye_image)

                    file.write(str(t_for_voxel_grid_timestamp)+ '\n')
                    # print(save_index )


                    save_index =  save_index +1







            # for frame in frame_list:
            #     out.write(frame)
            #
            #     # 释放VideoWriter对象
            # out.release()
            #
            # print("Video saved as", output_path)
#
# def process_image_wrapper(args):
#     user_name, ex_time = args
#     process_image(user_name, ex_time)
#
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#         args_list = [(user_name, ex_time) for user_name in range(1, 67) for ex_time in range(1, 7)]
#         executor.map(process_image_wrapper, args_list)
#
#
#

                    # print(x_y_warped_right_crop.shape)
                    # print(t_warped_crop_right.shape)
                    # print(p_warped_crop_right.shape)
                    # print(event_output_left.shape)

                    # event_output_left
                    # try:
                    #   print(x_y_warped_left_crop)
                    #   for (u, v) in x_y_warped_left_crop:
                    #
                    #     cv2.circle(img_warped_left, (u, v), 5, (0, 255, 0), -1)
                    #     cv2.imshow("img", img_warped_left)
                    #     cv2.waitKey(1000)
                    # except:
                    #     continue

                    # # 截取中心部分
                    # cropped_image = img_warped_left[yy:yy + height, xx:xx + width]
                    #
                    # # 使用cv2.imshow显示截取的部分
                    # cv2.imshow('Cropped Image', cropped_image)
                    # # cv2.waitKey(0)  # 显示图片并等待按键输入，0表示一直等待，直到按键按下
                    # cv2.imwrite('cropped_image.jpg', cropped_image)




                # cv2.waitKey(10000)



            # number_of_frames_to_insert = 19
            # main(
            #     root_event_folder,
            #     root_image_folder,
            #     root_output_folder,
            #     number_of_frames_to_insert,
            #     timestamp_output_folder,
            #     random_sequence_read_folder)
