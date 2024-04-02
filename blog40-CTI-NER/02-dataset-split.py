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
#自定义分隔符文本分割
def split_text(text):
    pattern = '###'
    nums = text.split(pattern) #获取字符的下标位置
    return nums
    
#-----------------------------------------------------------------------
#主函数
if __name__ == '__main__':
    #获取文件名
    path = "Mitre"
    savepath = "Mitre-Split"
    filenames = get_filepath(path)
    print(filenames)
    print("\n")

    #遍历文件内容
    k = 0
    begin = 1001  #命名计数
    while k<len(filenames):
        filename = "Mitre//" + filenames[k]
        print(filename)
        content = get_content(filename)
        print(content)

        #分割句子
        nums = split_text(content)

        #每隔五句输出为一个TXT文档
        n = 0
        result = ""
        while n<len(nums):
            if n>0 and (n%5)==0: #存储
                savename = savepath + "//" + str(begin) + "-" + filenames[k]
                print(savename)
                f = open(savename, "w", encoding="utf8")
                f.write(result)
                result = ""
                result = nums[n].lstrip() + "### "  #第一句
                begin += 1
                f.close()
            else:               #赋值
                result += nums[n].lstrip() + "### "
            n += 1
        k += 1
    

    
