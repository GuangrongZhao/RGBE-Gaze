# #利用正则表达式来找到每个中括号之间的东西
import re
# text = '<REC TIME="706.70380" TIME_TICK="94841050736" FPOGX="0.00000" FPOGY="0.00000" FPOGS="703.54175" FPOGD="0.02490" FPOGID="2275" FPOGV="0" LPOGX="0.43914" LPOGY="1.70982" LPOGV="0" RPOGX="0.52147" RPOGY="1.38244" RPOGV="0" BPOGX="0.43914" BPOGY="1.70982" BPOGV="0" LPCX="0.19850" LPCY="0.57422" LPD="40.11438" LPS="0.91299" LPV="0" RPCX="0.56451" RPCY="0.38077" RPD="25.59807" RPS="0.91299" RPV="0" LEYEX="0.04823" LEYEY="-0.01120" LEYEZ="0.51337" LPUPILD="0.00589" LPUPILV="1" REYEX="0.03413" REYEY="0.00632" REYEZ="0.53306" RPUPILD="0.00517" RPUPILV="1" BKID="0" BKDUR="0.00000" BKPMIN="24" LPMM="5.88584" LPMMV="0" RPMM="5.16697" RPMMV="0" PIXX="0.00000" PIXY="0.00000" PIXS="0.00000" PIXV="0" />'
#
# num = re.findall(r'".*?" ',text) #第一种方法，技巧是（\d）才是findall的返回值，注意findall返回的是list
# for i in range(0,len(num)):
#     print(num[i])
import matplotlib.pyplot as plt
import numpy as np
import csv
import concurrent.futures
from tqdm import tqdm

LPOGX = []
LPOGY = []
RPOGX = []
RPOGY = []
out = []
REYEX = []
REYEY = []
REYEZ = []
LEYEX = []
LEYEY = []
LEYEZ = []


def process_image(user_name, ex_time):
    print('user_name',user_name, 'ex_time',ex_time)
# for user_name in tqdm(range(5,9)):
#   for ex_time in tqdm(range(1,7)):


    inpath = 'D:/remote_dataset/user_%s/exp%s/gazepoint/gazepoint.txt'% (user_name, ex_time)
    outpath = 'D:/remote_dataset/user_%s/exp%s/gazepoint/gazepoint.csv'% (user_name, ex_time)

    fp = open(inpath, encoding= 'gb18030')
    lines = fp.readlines()  #读取全部内容 ，并以列表方式返回

    with open(outpath, "w", encoding="utf-8", newline="") as file:
     writer = csv.writer(file)
     for line in lines:
        # print(line)
        # break
        num = re.findall(r'".*?" ', line)  # 正则表达式，用来截取“”之间的字符串
        if len(num) != 0:
            # print(len(num))
            pass
        # if len(num)==21 or len(num)>30:
        # print(num)

        if "RPOGY" in line and "LPOGX" in line:
            wline = []
            # print(len(num))
            for j in range(len(num)):
                wline.append(num[j][1:-2])
            # print(wline)
            writer.writerow(wline)


def process_image_wrapper(args):
                user_name, ex_time = args
                process_image(user_name, ex_time)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args_list = [(user_name, ex_time) for user_name in range(10, 68) for ex_time in range(1,7)]
        executor.map(process_image_wrapper, args_list)


            # writer.writerow([num[8][1:-2]])
            # print(num[9][1:-2])
            # LPOGX.append(num[9][1:-2])
            # LPOGY.append(num[10][1:-2])
            # RPOGX.append(num[12][1:-2])
            # RPOGY.append(num[13][1:-2])

            # REYEX.append(num[33][1:-2])
            # REYEY.append(num[34][1:-2])
            # REYEZ.append(num[35][1:-2])
            # LEYEX.append(num[28][1:-2])
            # LEYEY.append(num[29][1:-2])
            # LEYEZ.append(num[30][1:-2])




    # out.append(num)

# print(len(LPOGX))

# x=list(np.arange(1, len(LPOGX)+1))
# print(LPOGX[1])
# plt.plot(LPOGX, LPOGY)
# plt.figure()
# plt.scatter(LPOGX, LPOGY, c='red', s=100, label='legend')
# plt.show()
# print(len(REYEX))
# print(REYEX)
# ft = open('RX.txt', 'w')
# for i in range(len(REYEX)):
#     ft.write(REYEX[i] + '\n')
#
# ft = open('RY.txt', 'w')
# for i in range(len(REYEX)):
#     ft.write(REYEY[i] + '\n')
#
# # print(len(LPOGX))
# ft = open('RZ.txt', 'w')
# for i in range(len(REYEX)):
#     ft.write(REYEZ[i] + '\n')

# ft = open('ry.txt', 'w')
# for i in range(1186):
#     ft.write(RPOGY[i] + '\n')