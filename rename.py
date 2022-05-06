import os
import sys
fipath="D:/Voxceleb1_train"
fiList = os.listdir(fipath)
for finame in fiList :
  filpath=fipath+"/"+finame
  filList = os.listdir(filpath)
  for filname in filList:
      filepath=filpath+"/"+filname
      fileList = os.listdir(filepath)
      print("start...")
# 得到进程当前工作目录
# 将当前工作目录修改为待修改文件夹的位置
# 名称变量
      num = 0000
# 遍历文件夹中所有文件
      currentpath = os.getcwd()
# 将当前工作目录修改为待修改文件夹的位置
      os.chdir(filepath)

      for fileName in fileList:
    # 文件重新命名
          os.rename(fileName, finame+"-"+filname+"-"+'%04d'%num+'.wav')
    # 改变编号，继续下一项
          num = num + 1

          print("end...")
# 改回程序运行前的工作目录
  os.chdir(currentpath)
# 刷新
  sys.stdin.flush()


