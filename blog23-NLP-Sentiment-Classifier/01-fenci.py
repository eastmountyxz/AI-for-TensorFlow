#encoding=utf-8  
import jieba  
  
text = "北京理工大学生前来应聘"  

data = jieba.cut(text,cut_all=True)   #全模式
print("[全模式]: ", " ".join(data))
  
data = jieba.cut(text,cut_all=False)  #精确模式  
print("[精确模式]: ", " ".join(data))
   
data = jieba.cut(text)  #默认是精确模式 
print("[默认模式]: ", " ".join(data))

data = jieba.cut_for_search(text)  #搜索引擎模式   
print("[搜索引擎模式]: ", " ".join(data))
