import os
import random

import cv2
import dlib
from imutils import face_utils
import numpy as np
import natsort
from tqdm import  tqdm
import h5py
import scipy.io
import concurrent.futures

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def normalizeData_face(img, face_model, landmarks, hr, ht, cam, total_landmark):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 400  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    total_num_point = total_landmark.shape[0]
    total_landmarks_normalized = cv2.perspectiveTransform(total_landmark, W)
    total_landmarks_normalized = total_landmarks_normalized.reshape(total_num_point, 2)

    return img_warped, landmarks_warped,total_landmarks_normalized


def normalizeData(img, face_model,landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (100, 60)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    # print(Fc.shape)
    le = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of left eye
    # print( 'le',le )
    # print('leshape', le.shape)
    # print('le21',le[1])
    le[0] = le[0]-2
    le[1] = le[1]-2
    re = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of right eye
    re[0] = re[0] + 2
    re[1] = re[1] - 2
    # print('re',re)
    ## normalize each eye
    data = []
    for et in [re, le]:
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et)  # actual distance between eye and original camera  %注意是针对眼睛的normalization

        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R,用它归一化gaze
        # print(R)

        # print(R)
        # R2rotation_vector = np.dot(S,R)   # 可能S 可以不要
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        # normalize the facial landmarks
        num_point = landmarks.shape[0]
        # print(landmarks.shape)
        landmarks_warped = cv2.perspectiveTransform(landmarks, W)

        landmarks_warped = landmarks_warped.reshape(num_point, 2)
        # print(R2rotation_vector)
        data.append([img_warped,landmarks_warped,S,R,hr_norm])

    return data
#



# user_name = 1
# ex_time = 5
def process_image(user_name, ex_time):
# print('user_name', user_name, 'ex_time', ex_time)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful

    with open('D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/headpose_r.txt' % (user_name, ex_time), "w") as headpose_r_file:
        with open('D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/headpose_t.txt' % (user_name, ex_time), "w") as headpose_t_file:
            with open('D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/no_detected_faces_index.txt' % (user_name, ex_time), "w") as no_detected_faces:
                                face_path = 'D:/remote_dataset/user_%s/exp%s/convert2eventspace/' % (user_name, ex_time)
                                # datanames = os.listdir(face_path)
                                datanames = [file for file in os.listdir(face_path) if file.endswith('.jpg')]
                                datanames = natsort.natsorted(datanames)
                                total_landmarks_normalized_list = [];
                                total_landmarks_list = [];
                                right_rv_gazenorm_list = [];
                                left_rv_gazenorm_list = [];

                                right_hr_norm_list = [];
                                left_hr_norm_list = [];

                                right_scale_vector_list = [];
                                left_scale_vector_list = [];
                                # with open('D:/remote_dataset/user_%s/exp%s/convert2eventspace/inter_time_align_gazepoint.txt' % (user_name, ex_time), 'r') as file:
                                #     # 逐行读取文件内容，并去除末尾的换行符
                                #     lines = file.readlines()
                                #     # 将每行数据转换为整数并存储在一个列表中
                                #     vec = [int(line.strip()) for line in lines]
                                #
                                # # 输出读取到的向量
                                # print(vec)
                                # vec = np.array(vec)-1
                                # for kv in tqdm(vec):

                                random.seed(1337)
                                # 生成一个包含1到10000之间所有整数的列表
                                # if ex_time == 5 or ex_time == 6:
                                #     all_numbers = list(range(1, 4001))
                                # else:
                                all_numbers = list(range(1,4001))
                                # 随机打乱这个列表
                                random.shuffle(all_numbers)
                                # 取前100个元素，即生成的不重复的随机数
                                # random_numbers = all_numbers[:200]
                                # if ex_time == 5 or ex_time == 6:
                                #     random_numbers = all_numbers[:200]
                                # else:
                                random_numbers = all_numbers[:100]

                                # 将结果排序
                                random_numbers.sort()

                                # 打印生成的数组
                                print(random_numbers)

                                valid_random_numbers = []

                                # random_file_path = r'D:\remote_dataset\processed_data\user_%s\exp%s\convert_frame_normalized\random_numbers.txt' % (
                                # user_name, ex_time)
                                #
                                # with open(random_file_path, 'r') as file:
                                #     random_filedata = file.readlines()

                                # 如果需要处理每行的数据
                                # for line in random_filedata:
                                #     # 在这里对每一行的数据进行处理
                                #     # 例如，你可以使用 split() 函数来拆分每一行的数据
                                #
                                #     # 示例：将每行的数据按空格分割成一个列表
                                #     values = line.split()
                                    # values 列表现在包含了每行的数据，你可以进一步处理它

                                # 如果需要将数据转化为数值，可以使用以下示例
                                # 示例：将数据行按空格分割并转化为浮点数
                                # parsed_data = [list(map(float, line.split())) for line in random_filedata]
                                # print(parsed_data[0][0])
                                #
                                # print(parsed_data )


                                for sv in tqdm(range(0,100)):

                                    # for sd in range(0,7):
                                    #     kv = int(parsed_data[sv][0])-3+sd # 这是为了gaze 360 来着 ,但是很可能有重复，因为随机不太行
                                        # print((datanames[kv]))
                                    # for kv in tqdm(range(0,len(datanames))):
                                    # for kv in tqdm(random_numbers):

                                    # kv = int(parsed_data[sv][0])
                                    # print(str(datanames[kv]))
                                    #输入图片
                                    # print(kv)
                                    kv = random_numbers[sv]
                                    img_file_name = 'D:/remote_dataset/user_%s/exp%s/convert2eventspace/%s' %(user_name, ex_time,str(datanames[kv]))
                                    # print('load input face image: ', img_file_name)
                                    image = cv2.imread(img_file_name)

                                    #landmark 提取所需要的文件

                                    detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
                                    if len(detected_faces) == 0:
                                        print('warning: reproduce face_detector')
                                        # no_detected_faces.write(str(kv) + '\n')
                                        # exit(0)
                                        # continue
                                        detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 3)  ## convert BGR image to RGB for dlib
                                    # print('detected one face')
                                    # print(detected_faces)
                                    if len(detected_faces) == 0:
                                        print('warning: no detected face')
                                        no_detected_faces.write(str(kv) + '\n')
                                        continue     # 跳过了当前循环

                                    valid_random_numbers.append(kv)
                                    shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
                                    shape = face_utils.shape_to_np(shape)
                                    landmarks = []
                                    for (x, y) in shape:
                                        landmarks.append((x, y))
                                    landmarks = np.asarray(landmarks)
                                    # print(landmarks.shape)
                                    # print(landmarks)


                                    # camera_matrix = np.array([[2415.48323379761 ,0 ,0],
                                    #                             [0 ,2416.44152477491, 0],
                                    #                              [731.768349768139 ,568.069547068642 ,1]])
                                    #
                                    camera_matrix = np.array([[1.712706065476114e+03, 0, 0],
                                                               [0 ,1713.40524379986, 0],
                                                            [6.560971442117219e+02 ,3.451634978903322e+02 ,1]])


                                    camera_matrix = camera_matrix.T
                                    camera_distortion = np.array([0,0,0,0,0])
                                    # print(camera_matrix)
                                    # print(camera_distortion)
                                    # print('estimate head pose')
                                    # load face model

                                    face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
                                    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
                                    face_model = face_model_load[landmark_use, :]
                                    facePts = face_model.reshape(6, 1, 3) # face_model 的对应点

                                    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]  # landmarks 提取器提取的点
                                    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
                                    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape

                                    landmarks = landmarks.astype(float)  # input to solvePnP function must be float type
                                    landmarks = landmarks.reshape(68, 1, 2 )  # input to solvePnP requires such shape

                                    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
                                    # print(hr)
                                    # print(str(np.squeeze(hr))[1:-2])
                                    headpose_r_file.write((str(np.squeeze(hr))[1:-1]) + '\n')
                                    headpose_t_file.write((str(np.squeeze(ht))[1:-1]) + '\n')

                                    # print(hr)
                                    # print(hr.shape)
                                    # print(ht)
                                    # print(ht.shape)

                                    # print('data normalization, i.e. crop the face image')
                                    img_normalized, landmarks_normalized, total_landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix,landmarks)  # face normalizeData
                                    landmarks_normalized = landmarks_normalized.astype(int)  # landmarks after data normalization
                                    total_landmarks_normalized = total_landmarks_normalized.astype(
                                        int)  # total_landmarks after data normalization

                                    total_landmarks = landmarks.astype(
                                        int)

                                    data = normalizeData(image, face_model, landmarks_sub, hr, ht, camera_matrix) # eye normalizeData

                                    # print(total_landmarks_normalized)
                                    # print(landmarks_normalized)

                                    # 绘制面部landmark
                                    # for (x, y) in total_landmarks_normalized:
                                    #     cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
                                    # cv2.imshow("img", img_normalized)
                                    # cv2.waitKey(1)
                                    # print(total_landmarks_normalized)
                                    # print(total_landmarks.reshape(68, 2))
                                    total_landmarks_normalized_list.append(total_landmarks_normalized)
                                    total_landmarks_list.append(total_landmarks.reshape(68, 2))

                                    # print(np.array(total_landmarks_normalized_list).shape)
                                    # Save the data to a MATLAB file


                                    # 面部的归一化图片
                                    output_path = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/face/%s' % (user_name, ex_time, str(datanames[kv]))
                                    # output_path = r'C:\Users\Aiot_Server\Desktop\remote_dataset\user_1\exp1\normalized\face\img_normalized.jpg'
                                    # print('save output image to: ', output_path)
                                    cv2.imwrite(output_path, img_normalized)

                                    for index,d in enumerate(data):
                                        img = d[0]
                                        lms = d[1].astype(int)
                                        scale_vector = d[2]
                                        R2rotation_vector = d[3]
                                        hr_norm = d[4]
                                        # R2rotation_vector = np.squeeze(R2rotation_vector)
                                        # print(R2rotation_vector)

                                        # print(lms)
                                        # # 绘制眼部landmark
                                        #
                                        # for (x, y) in lms:
                                        #     if 128 > x >= 0 and 0 <= y < 128:
                                        #         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                                        #         cv2.imshow("img",  img)
                                        #         cv2.waitKey(1)

                                        if index == 1:
                                            # 眼睛的归一化图片，先左眼后右眼
                                            output_path = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/%s' % (user_name, ex_time, str(datanames[kv]))
                                            # print('save output image to: ', output_path)
                                            cv2.imwrite(output_path, img)
                                            # left_rv_gazenorm.write((str(R2rotation_vector)[1:-1] )+ '\n')

                                            left_scale_vector_list.append(scale_vector)
                                            left_rv_gazenorm_list.append(R2rotation_vector)
                                            left_hr_norm_list.append(hr_norm)

                                        else:
                                            output_path = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/%s' % (user_name, ex_time, str(datanames[kv]))
                                            # print('save output image to: ', output_path)
                                            cv2.imwrite(output_path, img)
                                            # right_rv_gazenorm.write((str(R2rotation_vector)[1:-1] )+ '\n')

                                            right_scale_vector_list.append(scale_vector)
                                            right_rv_gazenorm_list.append(R2rotation_vector)
                                            right_hr_norm_list.append(hr_norm)

                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/left_hr_norm_list.mat' % (
                                        user_name, ex_time),
                                    {'left_hr_norm_list': np.array(
                                        left_hr_norm_list)})

                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/left_rv_gazenorm_list.mat' % (
                                        user_name, ex_time),
                                    {'left_rv_gazenorm_list': np.array(
                                        left_rv_gazenorm_list)})

                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/left_eye/left_scale_vector_list.mat' % (
                                        user_name, ex_time),
                                    {'left_scale_vector_list': np.array(
                                        left_scale_vector_list)})



                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/right_hr_norm_list.mat' % (
                                        user_name, ex_time),
                                    {'right_hr_norm_list': np.array(
                                        right_hr_norm_list)})


                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/right_rv_gazenorm_list.mat' % (
                                    user_name, ex_time),
                                    {'right_rv_gazenorm_list': np.array(
                                        right_rv_gazenorm_list)})

                                scipy.io.savemat(
                                    'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/right_eye/right_scale_vector_list.mat' % (
                                        user_name, ex_time),
                                    {'right_scale_vector_list': np.array(
                                        right_scale_vector_list)})


                                scipy.io.savemat('D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/face/total_landmarks_normalized_list.mat' %(user_name, ex_time),
                                                 {'total_landmarks_normalized_list': np.array(
                                                     total_landmarks_normalized_list)})


                                scipy.io.savemat('D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/face/total_landmarks_list.mat' %(user_name, ex_time),
                                                 {'total_landmarks_list': np.array(
                                                     total_landmarks_list)})


                                filename = 'D:/remote_dataset/processed_data_smooth/user_%s/exp%s/convert_frame_normalized/random_numbers.txt' % (
                                user_name, ex_time)

                                # 使用numpy.savetxt将数组保存到文本文件
                                np.savetxt(filename, np.array(valid_random_numbers), delimiter=' ')



def process_image_wrapper(args):
    user_name, ex_time = args
    process_image(user_name, ex_time)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args_list = [(user_name, ex_time) for user_name in range(1, 67) for ex_time in range(5,7)]
        executor.map(process_image_wrapper, args_list)
