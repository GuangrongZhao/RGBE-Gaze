# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2022 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# Acquisition.py shows how to acquire images. It relies on
# information provided in the Enumeration example. Also, check out the
# ExceptionHandling and NodeMapInfo examples if you haven't already.
# ExceptionHandling shows the handling of standard and Spinnaker exceptions
# while NodeMapInfo explores retrieving information from various node types.
#
# This example touches on the preparation and cleanup of a camera just before
# and just after the acquisition of images. Image retrieval and conversion,
# grabbing image data, and saving images are all covered as well.
#
# Once comfortable with Acquisition, we suggest checking out
# AcquisitionMultipleCamera, NodeMapCallback, or SaveToAvi.
# AcquisitionMulticpleCamera demonstrates simultaneously acquiring images from
# a number of cameras, NodeMapCallback serves as a good introduction to# programming with callbacks and events, and SaveToAvi exhibits video creation.
import matplotlib.pyplot as plt
import os
import PySpin as pyspin
import sys
from threading import Thread
import time
from pynput.keyboard import Key, Listener
import numpy as np
import cv2

import datetime

user_name = input("输入用户序号：")
ex_time = input("输入试验次数：")
ex_timelens = int(input("输入时间长度："))
path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/flir/'% (user_name, ex_time)
print(path)
framerate = 50
NUM_IMAGES = framerate * ex_timelens # number of images to grab
fileName = path + 'timestamp.txt'
fileName_win = path + 'timestamp_win.txt'
fileName_cpu = path + 'timestamp_cpu.txt'
processor = pyspin.ImageProcessor()
images_list = list()
images_file_list = []
import cv2
def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.delete:
        # Stop listener
        return False


def save_img():

    # while True:
    #     if len(images_list) > 0:
    for i in range(len(images_list)):
            # image_converted = processor.Convert(images_list[i], pyspin.PixelFormat_BGR8)
            image_converted = images_list[i].GetNDArray()
            # Create a unique filename
            # image_converted.Save(images_file_list[i])
            # del images_list[0], images_file_list[0]
            cv2.imwrite(images_file_list[i],image_converted)
        # else:3
        #     break
    print("finish*************************************************************************")
t_list = []
t_win_list = []
t_cpu_list = []
def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves 10 images from a device.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        #
        #  *** NOTES ***
        #  Because the example acquires and saves 10 images, setting acquisition
        #  mode to continuous lets the example finish. If set to single frame
        #  or multiframe (at a lower number of images), the example would just
        #  hang. This would happen because the example has been written to
        #  acquire 10 images while the camera would have been programmed to
        #  retrieve less than that.
        #
        #  Setting the value of an enumeration node is slightly more complicated
        #  than other node types. Two nodes must be retrieved: first, the
        #  enumeration node is retrieved from the nodemap; and second, the entry
        #  node is retrieved from the enumeration node. The integer value of the
        #  entry node is then set as the new value of the enumeration node.
        #
        #  Notice that both the enumeration and the entry nodes are checked for
        #  availability and readability/writability. Enumeration nodes are
        #  generally readable and writable whereas their entry nodes are only
        #  ever readable.
        #
        #  Retrieve enumeration node from nodemap

        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = pyspin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not pyspin.IsReadable(node_acquisition_mode) or not pyspin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not pyspin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images. Because the example calls for the
        #  retrieval of 10 images, continuous mode has been set.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        # cam.PixelFormat.SetValue(pyspin.PixelFormat_BGR8)

        cam.ExposureAuto.SetValue(pyspin.ExposureAuto_Off)
        print('Automatic exposure disabled...')
        exposure_time_to_set = 1000/framerate*1000
        exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure_time_to_set)
        print('Shutter time set to %s us...\n' % exposure_time_to_set)

        node_acquisition_framerate = pyspin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))

        if not pyspin.IsReadable(node_acquisition_framerate):
            print('Unable to retrieve frame rate. Aborting...')
            return False

        framerate_to_set = node_acquisition_framerate.GetValue()

        print('Frame rate to be set to %d...' % framerate_to_set)
        # cam.PixelFormat.SetValue(pyspin.PixelFormat_Mono8)
        # cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = pyspin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if pyspin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve, convert, and save images

        # Create ImageProcessor instance for post processing images

        ### Set Pixel Format to RGB8 ###


        # Set default image processor color processing method
        #
        # *** NOTES ***
        # By default, if no specific color processing algorithm is set, the image
        # processor will default to NEAREST_NEIGHBOR method.



        processor.SetColorProcessing(pyspin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        print('Enter the del :')
        with Listener(on_press=show) as listener:
            listener.join()
        t_win_list.append(time.time())
        t_cpu_list.append(cv2.getTickCount())
        # print('del well')
        cam.BeginAcquisition()
        t_win_list.append(time.time())
        t_cpu_list.append(cv2.getTickCount())
        for i in range(NUM_IMAGES):
            try:
                # print(i)
                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.

                image_result = cam.GetNextImage(1000) #a 64bit value that represents a timeout in milliseconds

                t_win = time.time()
                t = str((image_result.GetTimeStamp()))

                t_win_list.append(t_win)

                t_list.append(t)
                t_cpu_list.append(cv2.getTickCount())
                # print(t_list)
                # t = str((image_result.EventExposureEndTimestamp.GetValue()))

                #  Ensure image completion
                #
                #  *** NOTES ***
                #  Images can easily be checked for completion. This should be
                #  done whenever a complete image is expected or required.
                #  Further, check image status for a little more insight into
                #  why an image is incomplete.
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:

                    #  Print image information; height and width recorded in pixels
                    #
                    #  *** NOTES ***
                    #  Images have quite a bit of available metadata including
                    #  things such as CRC, image status, and offset values, to
                    #  name a few.
                    # width = image_result.GetWidth()
                    # height = image_result.GetHeight()
                    # print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8
                    #
                    #  *** NOTES ***
                    #  Images can be converted between pixel formats by using
                    #  the appropriate enumeration value. Unlike the original
                    #  image, the converted one does not need to be released as
                    #  it does not affect the camera buffer.
                    #
                    #  When converting images, color processing algorithm is an
                    #  optional parameter.
                    # image_converted = processor.Convert(image_result, pyspin.PixelFormat_Mono8)

                    # f.write(str(t) + "\n")
                    # f_win.write(str(t_win) + "\n")

                    if device_serial_number:
                        filename = path+'%d-%s.jpg' % (i, t)
                        # print(filename )
                    else:  # if serial number is empty
                        filename = path+'%d.jpg' % i
                        # print(filename)
                    image_data = processor.Convert(image_result, pyspin.PixelFormat_BGR8)
                    # print(image_result.GetFrameID())
                    # print(image_result)
                    # image_data = image_result.GetNDArray()

                    # cv2.namedWindow('a', 0)
                    # cv2.resizeWindow('a', 800, 600)  # 设置窗口大小
                    # cv2.imshow('a', image_data)
                    #
                    # # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # # Interval is in seconds.
                    # cv2.waitKey(1)
                    #
                    # print(image_data)

                    # Draws an image on the current figure
                    # plt.imshow(image_data, cmap='gray')
                    #
                    # # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # # Interval is in seconds.
                    # plt.pause(0.001)
                    #
                    # # Clear current reference of a figure. This will improve display speed significantly
                    # plt.clf()

                    images_list.append(image_data)

                    images_file_list.append(filename)

                    #  Save image
                    #  *** NOTES ***
                    #  The standard practice of the examples is to use device
                    #  serial numbers to keep images of one device from
                    #  overwriting those of another.
                    # image_converted.Save(filename)
                    # print('Image saved at %s' % filename)

                    #  Release image
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()

                    # print('')
                    # images_list.append(image_result)
                    # images_file_list.append(filename)
            except pyspin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

        with open(fileName, "w") as f:
            for i in t_list:
                f.write(str(i) + '\n')
        with open(fileName_win, "w") as f_win:
            for i in t_win_list:
                f_win.write(str(i) + '\n')
        with open(fileName_cpu, "w") as f_cpu:
            for i in t_cpu_list:
                f_cpu.write(str(i) + '\n')

    except pyspin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result




def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = pyspin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if pyspin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = pyspin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if pyspin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except pyspin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # task_Proc_save_img = Thread(target=save_img(images_list, images_file_list))
        # task_Proc_save_img.setDaemon(True)
        # task_Proc_save_img.start()
######################################################################################内存转存硬盘##############
        save_img()

        # Deinitialize camera
        cam.DeInit()
        print('s')
    except pyspin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = pyspin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    print(cam_list)
    # Run example on each camera


    for i, cam in enumerate(cam_list):
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        device_serial_number = ''
        node_device_serial_number = pyspin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if pyspin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)
        if device_serial_number == '21400529':
            print('Running example for camera %d...' % i)
            result &= run_single_camera(cam)
            print('Camera %d example complete... \n' % i)



    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()
    print('Done! Press Enter to exit...')
    # input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    # if main():
    #     sys.exit(0)
    # else:
    #     sys.exit(1)
    main()