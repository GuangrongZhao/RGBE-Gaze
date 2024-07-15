import cv2
from pynput import mouse
from pynput.keyboard import Key, Listener
import time
cpu_time = []


def on_click(x, y, button, pressed):


    if pressed and button == mouse.Button.left:
        f_cpu.write(str(cv2.getTickCount())+' '+str(x)+' ' + str(y) + "\n")
        print(x, y)
        # print(f"mouse Position: X={x}, Y={y}")
        # print(cv2.getTickCount())


        if ((time.time() - t_start) > ex_timelens) :


                   mouse_listener.stop()



def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.delete:
        # Stop listener
        return False

if __name__ == "__main__":

    user_name = input("Enter user number: ")
    ex_time = input("Enter experiment number: ")
    ex_timelens = int(input("Enter experiment duration: "))
    fileName_cpu = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/click_gt_cpu.txt' % (
    user_name, ex_time)
    print('Enter the del:')
    with Listener(on_press=show) as listener:
        listener.join()


    with open(fileName_cpu, "w") as f_cpu:
        t_start = time.time()


        mouse_listener = mouse.Listener(on_click=on_click)

        try:
            mouse_listener.start()


            mouse_listener.join()



        except ((time.time() - t_start) > ex_timelens) :


                   mouse_listener.stop()