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

    return img_warped, landmarks_warped,total_landmarks_normalized,S,R,hr_norm


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

# user_name = 1
# ex_time = 1

def process_image(user_name, ex_time):
    print('user_name', user_name, 'ex_time', ex_time)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful

    face_path = 'G:/remote_apperance_gaze_dataset/raw_data/user_%s/exp%s/convert2eventspace/' % (user_name, ex_time)
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

    scale_vector_list = [];
    rv_gazenorm_list = [];
    hr_norm_list = [];

    all_lines = []
    file_name = 'G:/remote_apperance_gaze_dataset/frame_eva_random_index/user_%s_exp_%s.txt' % (user_name, ex_time)

    with open(file_name, 'r') as file:
        # 读取每一行并添加到列表中
        for line in file:
            all_lines = np.append(all_lines, line.strip())  # 使用 strip() 方法去除末尾的换行符并添加到数组中

        # print(all_lines[0])
    # for kv in tqdm(range(0,len(datanames))):

    no_detected_faces = []
    for kv in tqdm(range(0,len(all_lines))):

        # print(all_lines[kv])
        # print( datanames[0])
        img_file_name = 'G:/remote_apperance_gaze_dataset/raw_data/user_%s/exp%s/convert2eventspace/%s' %(user_name, ex_time,str(datanames[ int(all_lines[kv])]))
        # print('load input face image: ', img_file_name)
        image = cv2.imread(img_file_name)
        # cv2.imshow("img", image)

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
            continue


        shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
        shape = face_utils.shape_to_np(shape)
        landmarks = []
        for (x, y) in shape:
            landmarks.append((x, y))
        landmarks = np.asarray(landmarks)

        camera_matrix = np.array([[1.712706065476114e+03, 0, 0],
                                   [0 ,1713.40524379986, 0],
                                [6.560971442117219e+02 ,3.451634978903322e+02 ,1]])


        camera_matrix = camera_matrix.T
        camera_distortion = np.array([0,0,0,0,0])

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



        img_normalized, landmarks_normalized, total_landmarks_normalized ,face_S,face_R,face_hr_norm= normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix,landmarks)  # face normalizeData
        landmarks_normalized = landmarks_normalized.astype(int)  # landmarks after data normalization
        total_landmarks_normalized = total_landmarks_normalized.astype(
            int)  # total_landmarks after data normalization

        total_landmarks = landmarks.astype(
            int)

        data = normalizeData(image, face_model, landmarks_sub, hr, ht, camera_matrix) # eye normalizeData



        total_landmarks_normalized_list.append(total_landmarks_normalized)
        total_landmarks_list.append(total_landmarks.reshape(68, 2))

        scale_vector_list.append(face_S)
        rv_gazenorm_list.append(face_R)
        hr_norm_list.append(face_hr_norm)

        output_path = 'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/%s' % (user_name, ex_time, str(datanames[ int(all_lines[kv])]))
        cv2.imwrite(output_path, img_normalized)

        for index,d in enumerate(data):
            img = d[0]
            lms = d[1].astype(int)
            scale_vector = d[2]
            R2rotation_vector = d[3]
            hr_norm = d[4]

            if index == 1:

                output_path = 'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/left_eye/%s' % (user_name, ex_time,  str(datanames[ int(all_lines[kv])]))

                cv2.imwrite(output_path, img)

                left_scale_vector_list.append(scale_vector)
                left_rv_gazenorm_list.append(R2rotation_vector)
                left_hr_norm_list.append(hr_norm)

            else:
                output_path = 'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/right_eye/%s' % (user_name, ex_time,  str(datanames[ int(all_lines[kv])]))
                # print('save output image to: ', output_path)
                cv2.imwrite(output_path, img)
                # right_rv_gazenorm.write((str(R2rotation_vector)[1:-1] )+ '\n')

                right_scale_vector_list.append(scale_vector)
                right_rv_gazenorm_list.append(R2rotation_vector)
                right_hr_norm_list.append(hr_norm)

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/left_eye/left_hr_norm_list.mat' % (
                user_name, ex_time),
            {'left_hr_norm_list': np.array(
                left_hr_norm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/left_eye/left_rv_gazenorm_list.mat' % (
                user_name, ex_time),
            {'left_rv_gazenorm_list': np.array(
                left_rv_gazenorm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/left_eye/left_scale_vector_list.mat' % (
                user_name, ex_time),
            {'left_scale_vector_list': np.array(
                left_scale_vector_list)})




        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/hr_norm_list.mat' % (
                user_name, ex_time),
            {'hr_norm_list': np.array(
                hr_norm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/rv_gazenorm_list.mat' % (
                user_name, ex_time),
            {'rv_gazenorm_list': np.array(
                rv_gazenorm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/scale_vector_list.mat' % (
                user_name, ex_time),
            {'scale_vector_list': np.array(
                scale_vector_list)})




        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/right_eye/right_hr_norm_list.mat' % (
            user_name, ex_time),
            {'right_hr_norm_list': np.array(
                right_hr_norm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/right_eye/right_rv_gazenorm_list.mat' % (
            user_name, ex_time),
            {'right_rv_gazenorm_list': np.array(
                right_rv_gazenorm_list)})

        scipy.io.savemat(
            'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/right_eye/right_scale_vector_list.mat' % (
                user_name, ex_time),
            {'right_scale_vector_list': np.array(
                right_scale_vector_list)})


        scipy.io.savemat('G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/total_landmarks_normalized_list.mat' %(user_name, ex_time),
                         {'total_landmarks_normalized_list': np.array(
                             total_landmarks_normalized_list)})


        scipy.io.savemat('G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/face/total_landmarks_list.mat' %(user_name, ex_time),
                         {'total_landmarks_list': np.array(
                             total_landmarks_list)})


        filename = 'G:/remote_apperance_gaze_dataset/processed_data/random_data_for_frame_method_eva/user_%s/exp%s/convert_frame_normalized/random_numbers.txt' % (
        user_name, ex_time)




def process_image_wrapper(args):
    user_name, ex_time = args
    process_image(user_name, ex_time)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args_list = [(user_name, ex_time) for user_name in range(1, 2) for ex_time in range(1,2)]
        executor.map(process_image_wrapper, args_list)
