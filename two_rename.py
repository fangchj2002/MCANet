import os
import sys
filepath="D:/Voxceleb1_train"
filList = os.listdir(filepath)
num =1
for fileName in filList:
    filpath = filepath + "/" + fileName
    fiList = os.listdir(filpath)
    print("start...")
# 得到进程当前工作目录
# 将当前工作目录修改为待修改文件夹的位置
# 名称变量

# 遍历文件夹中所有文件
    currentpath = os.getcwd()
# 将当前工作目录修改为待修改文件夹的位置
    os.chdir(filpath)
    for filName in fiList:

    # 文件重新命名
         os.rename(filName,'%04d'%num)
    # 改变编号，继续下一项
         num = num + 1
         print("end...")
# 改回程序运行前的工作目录
    print(num)
num = num

os.chdir(currentpath)
# 刷新
sys.stdin.flush()
