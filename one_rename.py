import os
import sys
filepath="D:/Voxceleb1_train"
filList = os.listdir(filepath)
num = 101
for fileName in filList:
    print("start...")
# 遍历文件夹中所有文件
    currentpath = os.getcwd()
# 将当前工作目录修改为待修改文件夹的位置
    os.chdir(filepath)
    # 文件重新命名
    os.rename(fileName,'%04d'%num)
    # 改变编号，继续下一项
    num = num + 1
    print("end...")
# 改回程序运行前的工作目录
print(num)
num = num

os.chdir(currentpath)
# 刷新
sys.stdin.flush()
