#encoding:utf-8
#By:Eastmount CSDN
import re
import os
import csv

#------------------------------------------------------------------------
#获取文件路径及名称
def get_filepath(path):
    entities = {}              #字段实体类别
    files = os.listdir(path)   #遍历路径
    return files

#-----------------------------------------------------------------------
#获取文件内容
def get_content(filename):
    content = ""
    fr = open(filename, "r", encoding="utf8")
    reader = csv.reader(fr)
    k = 0
    for r in reader:
        if k>0 and (r[0]!="" or r[0]!=" ") and r[1]!="":
            content += r[0] + " " + r[1] + "\n"
        elif (r[0]=="" or r[0]==" ") and r[1]!="":
            content += "UNK" + " " + r[1] + "\n"
        elif (r[0]=="" or r[0]==" ") and r[1]=="":
            content += "\n"
        k += 1
    return content
    
#-----------------------------------------------------------------------
#主函数
if __name__ == '__main__':
    #获取文件名
    #path = "train"
    #path = "test"
    path = "val"
    filenames = get_filepath(path)
    print(filenames)
    print("\n")
    #savefilename = "dataset-train.txt"
    #savefilename = "dataset-test.txt"
    savefilename = "dataset-val.txt"
    f = open(savefilename, "w", encoding="utf8")

    #遍历文件内容
    k = 0
    while k<len(filenames):
        filename = path + "//" + filenames[k]
        print(filename)
        content = get_content(filename)
        print(content)
        f.write(content)
        k += 1
    f.close()
