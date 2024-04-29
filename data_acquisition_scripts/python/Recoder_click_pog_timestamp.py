import cv2
from pynput import mouse
from pynput.keyboard import Key, Listener
import time
cpu_time = []


# 回调函数，用于处理鼠标点击事件
def on_click(x, y, button, pressed):


    if pressed and button == mouse.Button.left:
        f_cpu.write(str(cv2.getTickCount())+' '+str(x)+' ' + str(y) + "\n")
        print(x, y)
        # print(f"鼠标坐标: X={x}, Y={y}")
        # print(cv2.getTickCount())


        if ((time.time() - t_start) > ex_timelens) :

                # 用户按下 Ctrl+C 来停止程序
                   mouse_listener.stop()



def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.delete:
        # Stop listener
        return False

if __name__ == "__main__":

    user_name = input("输入用户序号：")
    ex_time = input("输入试验次数：")
    ex_timelens = int(input("输入时间长度："))
    fileName_cpu = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/click_gt_cpu.txt' % (
    user_name, ex_time)
    print('Enter the del:')
    with Listener(on_press=show) as listener:
        listener.join()


    with open(fileName_cpu, "w") as f_cpu:
        t_start = time.time()

        # 创建鼠标监听器
        mouse_listener = mouse.Listener(on_click=on_click)
        # 启动鼠标监听器
        try:
            mouse_listener.start()
            # 保持监听鼠标事件

            mouse_listener.join()



        except ((time.time() - t_start) > ex_timelens) :

                # 用户按下 Ctrl+C 来停止程序
                   mouse_listener.stop()