import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import natsort
from tqdm import  tqdm
import h5py

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

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
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

    return img_warped, landmarks_warped


def normalizeData(img, face_model,landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (100, 60)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of left eye

    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of right eye

    ## normalize each eye
    data = []
    for et in [re, le]:
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et)  # actual distance between eye and original camera

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
        R2rotation_vector, _ = cv2.Rodrigues(R)
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
        data.append([img_warped,landmarks_warped,R2rotation_vector])

    return data



if __name__ == '__main__':

    for user_name in range(1, 4):
        for ex_time in range(1, 4):

            left_rv_gazenorm_file = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/left_eye/rv_gazenorm.txt' % ( user_name, ex_time)

            right_rv_gazenorm_file = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/right_eye/rv_gazenorm.txt' % (user_name, ex_time)
            with open(left_rv_gazenorm_file , "w") as left_rv_gazenorm:
                with open(right_rv_gazenorm_file, "w") as right_rv_gazenorm:
                    with open('C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/headpose_r.txt' % (user_name, ex_time), "w") as headpose_r_file:
                        with open('C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/headpose_t.txt' % (user_name, ex_time), "w") as headpose_t_file:
                            face_path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/flir/' % (user_name, ex_time)
                            datanames = os.listdir(face_path)
                            datanames = natsort.natsorted(datanames)

                            for kv in tqdm(range(0,len(datanames)+1)):
                                # print(str(datanames[kv]))
                                #输入图片

                                img_file_name = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/flir/%s' %(user_name, ex_time,str(datanames[kv]))
                                # print('load input face image: ', img_file_name)
                                image = cv2.imread(img_file_name)

                                #landmark 提取所需要的文件
                                predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                                face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
                                detected_faces = face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1) ## convert BGR image to RGB for dlib
                                if len(detected_faces) == 0:
                                    print('warning: no detected face')
                                    exit(0)
                                # print('detected one face')
                                shape = predictor(image, detected_faces[0]) ## only use the first detected face (assume that each input image only contains one face)
                                shape = face_utils.shape_to_np(shape)
                                landmarks = []
                                for (x, y) in shape:
                                    landmarks.append((x, y))
                                landmarks = np.asarray(landmarks)

                                camera_matrix = np.array([[2415.48323379761 ,0 ,0],
                                                            [0 ,2416.44152477491, 0],
                                                             [731.768349768139 ,568.069547068642 ,1]])
                                camera_matrix = camera_matrix.T
                                camera_distortion = np.array( [0,0,0,0,0])
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
                                hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

                                headpose_r_file.write((str(np.squeeze(hr))[1:-2]) + '\n')
                                headpose_t_file.write((str(np.squeeze(ht))[1:-2]) + '\n')



                                # print(hr)
                                # print(hr.shape)
                                # print(ht)
                                # print(ht.shape)



                                # print('data normalization, i.e. crop the face image')
                                img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)  # face normalizeData
                                landmarks_normalized = landmarks_normalized.astype(int)  # landmarks after data normalization
                                data = normalizeData(image, face_model, landmarks_sub, hr, ht, camera_matrix) # eye normalizeData




                                # 绘制面部landmark
                                # for (x, y) in landmarks_normalized:
                                #     cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
                                # cv2.imshow("img", img_normalized)
                                # cv2.waitKey(1)

                                # 面部的归一化图片
                                output_path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/face/%s' % (user_name, ex_time, str(datanames[kv]))
                                # output_path = r'C:\Users\Aiot_Server\Desktop\remote_dataset\user_1\exp1\normalized\face\img_normalized.jpg'
                                # print('save output image to: ', output_path)
                                cv2.imwrite(output_path, img_normalized)


                                for index,d in enumerate(data):
                                    img = d[0]
                                    lms = d[1].astype(int)
                                    R2rotation_vector = d[2]

                                    R2rotation_vector = np.squeeze(R2rotation_vector)
                                    # print(R2rotation_vector.shape)

                                    # print( lms)
                                    # 绘制眼部landmark
                                    # for (x, y) in lms:
                                    #     if 128 > x >= 0 and 0 <= y < 128:
                                    #         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                                    #         print(x,y)

                                    if index == 1:
                                        # 眼睛的归一化图片，先左眼后右眼
                                        output_path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/left_eye/%s' % (user_name, ex_time, str(datanames[kv]))
                                        # print('save output image to: ', output_path)
                                        cv2.imwrite(output_path, img)
                                        left_rv_gazenorm.write((str(R2rotation_vector)[1:-2] )+ '\n')
                                        # cv2.imshow("img",  img)
                                        # cv2.waitKey(1)
                                    else:
                                        output_path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/normalized/right_eye/%s' % (user_name, ex_time, str(datanames[kv]))
                                        # print('save output image to: ', output_path)
                                        cv2.imwrite(output_path, img)
                                        right_rv_gazenorm.write((str(R2rotation_vector)[1:-2] )+ '\n')