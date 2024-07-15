######################################################################################
# GazepointAPI.py - Example Client
# Written in 2013 by Gazepoint www.gazept.com
#
# To the extent possible under law, the author(s) have dedicated all copyright 
# and related and neighboring rights to this software to the public domain worldwide. 
# This software is distributed without any warranty.
#
# You should have received a copy of the CC0 Public Domain Dedication along with this 
# software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
######################################################################################
import cv2
import socket
import time
import threading
from pynput.keyboard import Key, Listener


# Host machine IP
HOST = '127.0.0.1'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)
data_list_series = []
win_time = []
cpu_time = []
def func1():
    ft = open(path+'gazepoint.txt', 'w')
    for i in data_list_series:
            ft.write(str(i) + '\n')

def func2():
    fx = open(path+'time_win.txt', 'w')
    for i in win_time:
            fx.write(str(i) + '\n')

def func3():
    fcpu = open(path+'time_cpu.txt', 'w')
    for i in cpu_time:
            fcpu.write(str(i) + '\n')


def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.delete:
        # Stop listener
        return False



user_name = input("Enter user number: ")
ex_time = input("Enter experiment number: ")
ex_timelens = int(input("Enter experiment duration: "))

path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/gazepoint/'% (user_name, ex_time)
print(path)



print('Enter the del :')
with Listener(on_press=show) as listener:
    listener.join()# print('del well')
cpu_time.append(cv2.getTickCount())
start_time = time.time()
# print(start_time)
sss = 1
k = 1


win_time.append(start_time) # Also use the second timestamp for alignment


# Connect to Gazepoint API
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(ADDRESS)

# Send commands to initialize data streaming
s.send(str.encode('<SET ID="ENABLE_SEND_COUNTER" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n'))  # TIME The time elapsed in seconds since the last system initialization or calibration.
s.send(str.encode('<SET ID="ENABLE_SEND_TIME_TICK" STATE="1" />\r\n'))  # Indicates the number of CPU time ticks, used for high-precision synchronization with other data collected on the same CPU.
s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
# Only focus on the fix state
# FPOGX, FPOGY (0,0) is at the top left, (0.5, 0.5) is the center of the screen, (1.0, 1.0) is at the bottom right.
# FPOGS The start time of the fixed POG since system initialization or calibration, in seconds.
# FPOGD The duration of the POG fixation, in seconds.
# FPOGV Valid flag indicating whether the fixed POG data is valid, with a value of 1 (TRUE) if valid, and 0 (FALSE) if not.
# FPOGV is only truly valid when one or both eyes are detected and a fixation is detected.
# FPOGV is false at other times, such as when the subject blinks, when there is no face in the field of view, when the eyes move to the next fixation point (i.e., saccade).
s.send(str.encode('<SET ID="ENABLE_SEND_POG_LEFT" STATE="1"/>\r\n'))
# LPOGX, LPOGY X and Y coordinates of the left eye POG as a fraction of screen size, valid flag, LPOGV value is 1 if data is valid, and 0 if not.
s.send(str.encode('<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1"/>\r\n'))
# RPOGX, RPOGY X and Y coordinates of the right eye POG as a fraction of screen size, valid flag, RPOGV value is 1 if data is valid, and 0 if not.
s.send(str.encode('<SET ID="ENABLE_SEND_POG_BEST" STATE="1"/>\r\n'))
# BPOGX, BPOGY The "best value" of the left/right eye POG data, which is the average of the left and right eye POGs if both are available, or if neither is available, it is the left or right eye depending on which one is valid.
# BPOGV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.
# For most applications, use FPOG, which includes the fixation filtering of BPOG data.
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1"/>\r\n'))
# LPCX, LPCY X and Y coordinates of the left eye pupil in the camera image, as a fraction of the camera image size.
# LPD Diameter of the left eye pupil, in pixels.
# LPS Scale factor for the left eye pupil (unitless). Value equals 1 when calibrated at depth, less than 1 when the user is closer to the eye tracker, and greater than 1 when the user is farther away.
# LPV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1"/>\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1"/>\r\n'))
# LEYEX, LEYEY, LEYEZ X, Y, and Z coordinates of the left eye relative to the camera focal point, in meters.
# LPUPILD Diameter of the left eye pupil, in meters.
# LPUPILV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1"/>\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_BLINK" STATE="1"/>\r\n'))
# BKID Each blink is assigned an ID value, which increments by 1. For records where no blink is detected, the BKID value is 0.
# BKDUR Duration of the previous blink, in seconds.
# BKPMIN Number of blinks in the previous 60 seconds.
s.send(str.encode('<SET ID="ENABLE_SEND_PUPILMM" STATE="1"/>\r\n'))
# LPMM Diameter of the left eye pupil, in millimeters.
# LPMMV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.
# RPMM Diameter of the right eye pupil, in millimeters.
# RPMMV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.
s.send(str.encode('<SET ID="ENABLE_SEND_PIX" STATE="1"/>\r\n'))
# PIXX, PIXY X and Y coordinates of the marker in the camera image, as a fraction of the camera image size.
# PIXS Scale conversion factor for converting pixels (such as pupil size) to millimeters. Multiply values in pixels by PIXS to convert to millimeters.
# PIXV Valid flag indicating whether the data is valid, with a value of 1 if valid, and 0 if not.

s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))


# data_list_series.append(start_time)
while 1:
    # Receive datac
    rxdat = s.recv(1024)
    data = bytes.decode(rxdat)

    # if sss>10 :    #  if k == 1 :
    save_time = time.time()

    cpu_time.append(cv2.getTickCount())
    win_time.append(save_time)
    # k = k + 1
    data_list_series.append(data)
    sss = sss + 1
    if (time.time() - start_time) > ex_timelens:
        # print(time.time())
        break

s.close()

print(data_list_series)


# timestamp = 0
# FPOGX = 0
# FPOGY = 0
# FPOGV = 0
# CX = 0
# CY = 0
#
# # Split data string into a list of name="value" substrings
# datalist = data.split(" ")
#
# # Iterate through list of substrings to extract data values
# for el in datalist:
#     if (el.find("FPOGX") != -1):
#         FPOGX = float(el.split("\"")[1])
#
#     if (el.find("FPOGY") != -1):
#         FPOGY = float(el.split("\"")[1])
#
#     if (el.find("FPOGV") != -1):
#         FPOGV = float(el.split("\"")[1])
#
#     if (el.find("CX") != -1):
#         CX = float(el.split("\"")[1])
#
#     if (el.find("CY") != -1):
#         CY = float(el.split("\"")[1])

# # Print results
# # print("FPOGD:", FPOGD)
# print("FPOGX:", FPOGX)
# print("FPOGX:", FPOGX)
# print("FPOGY:", FPOGY)
# print("FPOGV:", FPOGV)
# print("CX:", CX)
# print("CY:", CY)
# print("\n")

t1 = threading.Thread(target=func1)
t2 = threading.Thread(target=func2)
t3 = threading.Thread(target=func3)


t1.start()
t2.start()
t3.start()




t1.join()
t2.join()
t3.join()


print('finish')


#右眼：2,6,9