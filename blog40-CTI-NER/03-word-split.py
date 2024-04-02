#encoding:utf-8
#By:Eastmount CSDN
import re
import os

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
    with open(filename, "r", encoding="utf8") as f:
        for line in f.readlines():
            content += line.replace("\n"," ")
    return content
            
#---------------------------------------------------------------------
#空格分隔获取英文单词
def split_word(text):
    nums = text.split(" ")
    #print(nums)
    return nums

#-----------------------------------------------------------------------
#主函数
if __name__ == '__main__':
    #获取文件名
    path = "Mitre-Split"
    savepath = "Mitre-Split-Word"
    filenames = get_filepath(path)
    print(filenames)
    print("\n")

    #遍历文件内容
    k = 0
    while k<len(filenames):
        filename = path + "//" + filenames[k]
        print(filename)
        content = get_content(filename)
        content = content.replace("###","\n")

        #分割句子
        nums = split_word(content)
        #print(nums)
        savename = savepath + "//" + filenames[k]
        f = open(savename, "w", encoding="utf8")
        for n in nums:
            if n != "":
                #替换标点符号
                n = n.replace(",", "")
                n = n.replace(";", "")
                n = n.replace("!", "")
                n = n.replace("?", "")
                n = n.replace(":", "")
                n = n.replace('"', "")
                n = n.replace('(', "")
                n = n.replace(')', "")
                n = n.replace('’', "")
                n = n.replace('\'s', "")
                #替换句号
                if ("." in n) and (n not in ["U.S.","U.K."]):
                    n = n.rstrip(".")
                    n = n.rstrip(".\n")
                    n = n + "\n"
                f.write(n+"\n")
        f.close()
        k += 1
