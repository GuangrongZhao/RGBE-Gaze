import cv2
from pynput import mouse
from pynput.keyboard import Key, Listener
import time
cpu_time = []



def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.f5:
        # Stop listener
        return False

if __name__ == "__main__":

    user_name = input("输入用户序号：")
    ex_time = input("输入试验次数：")
    fileName_cpu = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/f5_gt_cpu.txt' % (
    user_name, ex_time)

    with open(fileName_cpu, "w") as f_cpu:
        print('等待F5:')
        with Listener(on_press=show) as listener:
            listener.join()

            f_cpu.write(str(cv2.getTickCount()) + "\n")
