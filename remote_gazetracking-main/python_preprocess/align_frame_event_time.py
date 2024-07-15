import numpy as np

'''
This step does the event and frame event alignment and generates the corresponding timestamp for the frame starting from 0
'''

for user_num in range(1,67):
        for exp_num in [1,2,3,4,5,6]:
            print('user_num',user_num  )
            print('exp_num',exp_num )
            root_path = 'G:/remote_apperance_gaze_dataset/raw_data/user_'+str(user_num)+'/exp'+str(exp_num)

           
            with open(root_path+"/prophesee/event_win.txt") as eventf:
                event_timestamp=[float(i) for i in eventf.readlines()]
                
            with open(root_path+"/convert2eventspace/timestamp_win.txt") as framef:
                frame_timestamp=[float(i) for i in framef.readlines() ]

            
            event_start = event_timestamp[1]* 10 ** 7
            frame_start = frame_timestamp[1]* 10 ** 7
            

            frame_time = np.loadtxt(root_path + '/convert2eventspace/timestamp.txt').reshape((-1, 1))
            frame_time = frame_time/100
            this_start = frame_start-event_start
            print(this_start)
           
            frame_time_start = frame_time[0].copy()

            for t in range(len(frame_time)):
                frame_time[t] = (frame_time[t] - frame_time_start + this_start) / 10+20*1000

            np.savetxt('G:/remote_apperance_gaze_dataset/processed_data/random_data_for_event_method_eva/frame_align_event_timestamp/user_'+str(user_num)+'_exp_'+str(exp_num)+'.txt', frame_time, fmt="%i")
            # np.savetxt('/home/sduu2/userspace-18T-2/remote_apperance_gaze_dataset/event_interpolation_samples/user_'+str(user_num)+'/exp'+str(exp_num)+'/frame_align_event_timestamp_difference.txt',[str(this_start)], fmt="%s")