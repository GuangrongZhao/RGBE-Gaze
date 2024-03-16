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



user_name = input("输入用户序号：")
ex_time = input("输入试验次数：")
ex_timelens = int(input("输入时间长度："))

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


win_time.append(start_time) # 也用第二个时间戳对齐


# Connect to Gazepoint API
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(ADDRESS)

# Send commands to initialize data streaming
# s.send(str.encode('<SET ID="ENABLE_SEND_COUNTER" STATE="1" />\r\n'))
#The time elapsed in seconds since the last system initialization or calibration.

s.send(str.encode('<SET ID="ENABLE_SEND_COUNTER" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n')) # TIME 自上次系统初始化或校准以来所经过的时间，以秒为单位。
s.send(str.encode('<SET ID="ENABLE_SEND_TIME_TICK" STATE="1" />\r\n')) # 表示CPU的时间刻度数，用于与同一CPU上收集的其他数据进行高精度同步。
s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
# 只关注fix的状态
# FPOGX, FPOGY （0,0）位于左上角，（0.5、0.5）是屏幕中心，（1.0、1.0）位于右下角。
# FPOGS 自系统初始化或校准以来，固定POG的开始时间，以秒为单位。
# FPOGD 注视的持续时间为POG，以秒为单位。
# FPOGV 如果固定POG数据有效，则为值为1（TRUE）的有效标志，如果固定POG数据不有效，则为0（FALSE）。
# 只有当检测到其中一只或两个眼睛并检测到注视时，FPOGV才真实有效。
# FPOGV 在其他时候都是假的，例如，当受试者眨眼时，当视野中没有脸时，当眼睛移动到下一个注视点时（即扫视）。
s.send(str.encode('<SET ID="ENABLE_SEND_POG_LEFT" STATE="1"/>\r\n'))
# LPOGX, LPOGY 左眼POG的X和Y坐标，作为屏幕大小的一部分，有效标志，LPOGV 如果数据是有效的，其值为1，如果不是，则为0。
s.send(str.encode('<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1"/>\r\n'))
# RPOGX, RPOGY 右眼POG的X和Y坐标，作为屏幕大小的一部分，有效标志，RPOGV 如果数据是有效的，其值为1，如果不是，则为0。
s.send(str.encode('<SET ID="ENABLE_SEND_POG_BEST" STATE="1"/>\r\n'))
# BPOGX, BPOGY 左/右眼POG数据的“最佳值”，如果两者都可用，即左眼和右眼POG的平均值，或者如果都没有，就是左眼或右眼，这取决于哪一个是有效的。
# BPOGV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0。
# 对于大多数应用程序，请使用FPOG，其中包括对BPOG数据的固定过滤。
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1"/>\r\n'))
# LPCX, LPCY 相机图像中左眼瞳孔的X坐标和Y坐标，作为相机图像大小的一小部分。
# LPD 左眼瞳孔的直径，以像素为单位。
# LPS 左眼瞳孔的比例因子（无单位）。在校准深度时值等于1，当用户更接近眼动仪时值小于1，当用户更远时值大于1。
# LPV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0。
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1"/>\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1"/>\r\n'))
# LEYEX, LEYEY, LEYEZ 左眼相对于相机焦点的X、Y和Z坐标，以米为单位
# LPUPILD 左眼瞳孔的直径，以米为单位
# LPUPILV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1"/>\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_BLINK" STATE="1"/>\r\n'))
# BKID 每次眨眼都被分配一个ID值，并增加1。对于未检测到闪烁的记录，BKID值为0。
# BKDUR 前一次眨眼的持续时间，以秒为单位。
# BKPMIN 前60秒时间内眨眼的次数。
s.send(str.encode('<SET ID="ENABLE_SEND_PUPILMM" STATE="1"/>\r\n'))
#  LPMM  左眼瞳孔的直径，以毫米为单位。
#  LPMMV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0。
#  RPMM 右眼瞳孔的直径，以毫米为单位。
#  RPMMV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0。
s.send(str.encode('<SET ID="ENABLE_SEND_PIX" STATE="1"/>\r\n'))
# PIXX, PIXY 相机图像中标记的X和Y坐标，作为相机图像大小的一部分。
# PIXS 从像素（如瞳孔大小）转换为毫米的比例转换因子。 将以像素为单位的值乘以 PIXS 以转换为毫米。
# PIXV 如果数据有效，则为值为1的有效标志，如果数据无效，则为0。

# s.send(str.encode('<GET ID="TIME_TICK_FREQUENCY" />>\r\n'))

s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))

# path = r'C:/Users/Aiot_Server/Desktop/remote_dataset/eyetracker2world/'



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

# 等待所有线程执行完毕


t1.join()
t2.join()
t3.join()


print('finish')


#右眼：2,6,9
