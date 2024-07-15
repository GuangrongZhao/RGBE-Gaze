# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision SDK to visualize events from a live camera or a RAW file
"""
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
from  metavision_sdk_cv import TrailFilterAlgorithm
import argparse
import os
import time
import threading
from pynput.keyboard import Key, Listener
from metavision_hal import I_LL_Biases,DeviceDiscovery
from datetime import datetime
# print(str(datetime.utcnow()))
import cv2

t_list = []
x_list = []
y_list = []
p_list = []
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


user_name = input("Enter user number: ")
ex_time = input("Enter experiment number: ")
ex_timelens = int(input("Enter experiment duration: "))

path = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/prophesee/'% (user_name, ex_time)
fileName = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/prophesee/event_win.txt'% (user_name, ex_time)
fileName_cpu = 'C:/Users/Aiot_Server/Desktop/remote_dataset/user_%s/exp%s/prophesee/event_cpu.txt'% (user_name, ex_time)
print(path)



def func1():
    ft = open(path+'t.txt', 'w')
    for i in t_list:
        for j in i:
            ft.write(str(j) + '\n')

def func2():
    fx = open(path+'x.txt', 'w')
    for i in x_list:
        for j in i:
            fx.write(str(j) + '\n')

def func3():
    fy = open(path+'y.txt', 'w')
    for i in y_list:
        for j in i:
            fy.write(str(j) + '\n')

def func4():
    fp = open(path+'p.txt', 'w')
    for i in p_list:
        for j in i:
            fp.write(str(j) + '\n')
def show(key):
    # print('\nYou Entered {0}'.format( key))
    if key == Key.delete:
        # Stop listener
        return False

def main():
    """ Main """
    args = parse_args()
    evs_list = []
    # Events iterator on Camera or RAW file


    # height, width = mv_iterator.get_size()  # Camera Geometry
    # print(height)
    # print(width)

    # Helper iterator to emulate realtime
    # if not is_live_camera(args.input_path):
    #     mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # # Window - Graphical User Interface
    # with MTWindow(title="Metavision Events Viewer", width=width, height=height,
    #               mode=BaseWindow.RenderMode.BGR) as window:
    #     def keyboard_cb(key, scancode, action, mods):
    #         if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
    #             window.set_close_flag()
    #
    #     window.set_keyboard_callback(keyboard_cb)
    #
    #     # Event Frame Generator
    #     event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
    #                                                        palette=ColorPalette.Dark)
    #
    #     def on_cd_frame_cb(ts, cd_frame):
    #         window.show_async(cd_frame)
    #
    #     event_frame_gen.set_output_callback(on_cd_frame_cb)

    # print(str(datetime.utcnow()))

    # Process events

    device = DeviceDiscovery.open('')
    if not device:
        print("Could not open camera. Make sure you have an event-based device plugged in")
        return 1

    # bias_file = args.bias_file
    # if bias_file:
    i_ll_biases = device.get_i_ll_biases()


    if i_ll_biases is not None:
        print(i_ll_biases.get_all_biases())
    # i_ll_biases.set("bias_fo", -35)
    i_ll_biases.set("bias_diff_off", 20)
    # i_ll_biases.set("bias_diff_on", 140)
    # i_ll_biases.set("bias_refr", 235)
    print(i_ll_biases.get_all_biases())
    print("hhh")



    # i_ll_biases.set("bias_fo", 55)
    # i_ll_biases.set("bias_hpf", 0)
    # i_ll_biases.set("bias_diff_off", 0)
    # i_ll_biases.set("bias_diff_on", 0)
    # i_ll_biases.set("bias_refr", 235)

    # print(i_ll_biases.get_all_biases())
    # erc=device.get_i_erc()
    # erc.enable(False)
    # print(erc.is_enabled())
    # print(erc.get_cd_event_rate())
    # ficker = device.get_i_antiflicker_module()
    # ficker.disable()
    # # ev_rate=device.get_i_event_rate()
    # # ev_rate.enable(False)
    # noise=device.get_i_noisefilter_module()
    # noise.disable()
    # print(erc.is_enabled())

    # print("xxx")

    # print(device.I_LL_Biases.get_all_biases())
    # Events iterator on Camera or RAW file
    # mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
    with open(fileName_cpu, "w") as f_cpu:
        with open(fileName, "w") as f:
            j = 0
            # if j == 0:
                # .timestamp() * 1000
                # j = j + 1

            print('Enter the del:')
            with Listener(on_press=show) as listener:
                listener.join()
            # mv_iterator = EventsIterator(input_path=args.input_path, delta_t = 1000, relative_timestamps = False)
            t_start_before = time.time()
            cpu_start_before = cv2.getTickCount()
            mv_iterator = EventsIterator.from_device(device=device)
            t_start = time.time()
            cpu_start = cv2.getTickCount()
            for evs in mv_iterator:
                # Dispatch system events to the window
                # print(len(evs['t']))
                # print(evs['t'])
               if evs.size != 0:
                   if j == 0:
                       first_event = time.time()
                       first_event_cpu = cv2.getTickCount()
                       j = j + 1

                   # t, x, y, p = evs['t'], evs['x'], evs['y'], evs['p']
                   # print(len(t))
                   # print(t)
                   # print(evs['t'])
                   # print(t[0])
                   # print(evs['t'][0])
                   # # print('s')
                   # t_list.append(t)
                   # print(t_list[0][0])
                   # print('s')               # break
                   # print(len(t_list))
                   # x_list.append(x)
                   # print(x[-1])
                   # print('s')
                   # print(x_list[-1])

                   t_list.append(evs['t'])
                   x_list.append(evs['x'])
                   y_list.append(evs['y'])
                   p_list.append(evs['p'])

               if ((time.time() - t_start) > ex_timelens):
                  end_events = time.time()
                  end_event_cpu = cv2.getTickCount()
                  break
            # print(len(t_list))

            f_cpu.write(str(cpu_start_before) + "\n")
            f_cpu.write(str(cpu_start) + "\n")
            f_cpu.write(str(first_event_cpu) + "\n")
            f_cpu.write(str(end_event_cpu) + "\n")

            f.write(str(t_start_before) + "\n")
            f.write(str(t_start) + "\n")
            f.write(str(first_event) + "\n")
            f.write(str(end_events) + "\n")

        # evs_list.append(evs)
        # print(len(evs_list))
            # EventLoop.poll_and_dispatch()
            # event_frame_gen.process_events(evs)

            # if window.should_close():
            #     break


    t1 = threading.Thread(target=func1)
    t2 = threading.Thread(target=func2)
    t3 = threading.Thread(target=func3)
    t4 = threading.Thread(target=func4)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    # 等待所有线程执行完毕
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    print('finish')


if __name__ == "__main__":
    main()
