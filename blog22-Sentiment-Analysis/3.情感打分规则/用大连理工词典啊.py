# coding: utf-8
import sys
import gzip
from collections import defaultdict
from itertools import product
import jieba
import csv
import pandas as pd

class Struct(object):
    def __init__(self, word, sentiment, pos,value, class_value):
        self.word = word
        self.sentiment = sentiment
        self.pos = pos
        self.value = value
        self.class_value = class_value

class Result(object):
    def __init__(self,score, score_words,not_word, degree_word ):
        self.score = score
        self.score_words = score_words
        self.not_word = not_word
        self.degree_word = degree_word

class Score(object):
        # 七个情感大类对应的小类简称: 尊敬
        score_class = {'乐':['PA','PE'],
                       '好':['PD','PH', 'PG','PB','PK'],
                       '怒':['NA' ],
                       '哀':['NB','NJ','NH', 'PF'],
                       '惧':['NI', 'NC', 'NG'],
                       '恶':['NE', 'ND', 'NN','NK','NL'],
                       '惊':['PC']
                       }
        # 大连理工大学 -> ICTPOS 3.0
        POS_MAP = {
            'noun': 'n',
            'verb': 'v',
            'adj': 'a',
            'adv': 'd',
            'nw': 'al',  # 网络用语
            'idiom': 'al',
            'prep': 'p',
        }

        # 否定词
        NOT_DICT = set(['不','不是','不大', '没', '无', '非', '莫', '弗', '毋',
                        '勿', '未', '否', '别', '無', '休'])

        def __init__(self, sentiment_dict_path, degree_dict_path, stop_dict_path ):
            self.sentiment_struct,self.sentiment_dict = self.load_sentiment_dict(sentiment_dict_path)
            self.degree_dict = self.load_degree_dict(degree_dict_path)
            self.stop_words = self.load_stop_words(stop_dict_path)

        def load_stop_words(self, stop_dict_path):
            stop_words = [w for w in open(stop_dict_path).readlines()]
            #print (stop_words[:100])
            return stop_words

        def remove_stopword(self, words):
            words = [w for w in words if w not in self.stop_words]
            return words

        def load_degree_dict(self, dict_path):
            """读取程度副词词典
            Args:
                dict_path: 程度副词词典路径. 格式为 word\tdegree
                           所有的词可以分为6个级别，分别对应极其, 很, 较, 稍, 欠, 超
           Returns:
                返回 dict = {word: degree}
            """
            degree_dict = {}
            with open(dict_path, 'r', encoding='UTF-8') as f:
                for line in f:
                    line = line.strip()
                    word, degree = line.split('\t')
                    degree = float(degree)
                    degree_dict[word] = degree
            return degree_dict

        def load_sentiment_dict(self, dict_path):
            """读取情感词词典
            Args:
                dict_path: 情感词词典路径. 格式请看 README.md
            Returns:
                返回 dict = {(word, postag): 极性}
            """
            sentiment_dict = {}
            sentiment_struct = []

            with open(dict_path, 'r', encoding='UTF-8') as f:
            #with gzip.open(dict_path) as f:
                for index, line in enumerate(f):
                    if index == 0:  # title,即第一行的标题
                        continue
                    items = line.split('\t')
                    word = items[0]
                    pos = items[1]
                    sentiment=items[4]
                    intensity = items[5]  # 1, 3, 5, 7, 9五档, 9表示强度最大, 1为强度最小.
                    polar = items[6]      # 极性
                    
                    # 将词性转为 ICTPOS 词性体系
                    pos = self.__class__.POS_MAP[pos]
                    intensity = int(intensity)
                    polar = int(polar)

                    # 转换情感倾向的表现形式, 负数为消极, 0 为中性, 正数为积极
                    # 数值绝对值大小表示极性的强度 // 分成3类，极性：褒(+1)、中(0)、贬(-1)； 强度为权重值
                    value = None
                    if polar == 0:            # neutral
                        value = 0
                    elif polar == 1:          # positive
                        value = intensity
                    elif polar == 2:          # negtive
                        value = -1 * intensity
                    else:  # invalid
                        continue

                    #key = (word, pos, sentiment )
                    key = word
                    sentiment_dict[key] = value

                    #找对应的大类
                    for item in self.score_class.items():
                        key = item[0]
                        values = item[1]
                        #print(key)
                        #print(value)
                        for x in values:
                            if (sentiment==x):
                                class_value = key # 如果values中包含，则获取key
                    sentiment_struct.append(Struct(word, sentiment, pos,value, class_value))
            return  sentiment_struct, sentiment_dict

        def findword(self, text): #查找文本中包含哪些情感词
            word_list = []
            for item in self.sentiment_struct:
                if item.word in text:
                    word_list.append(item)
            return word_list

        def classify_words(self, words):
            # 这3个键是词的序号(索引)
            
            sen_word = {}                 
            not_word = {}
            degree_word = {}
            # 找到对应的sent, not, degree;      words 是分词后的列表
            for index, word in enumerate(words):
                if word in self.sentiment_dict and word not in self.__class__.NOT_DICT and word not in self.degree_dict:
                    sen_word[index] = self.sentiment_dict[word]
                elif word in self.__class__.NOT_DICT and word not in self.degree_dict:
                    not_word[index] = -1
                elif word in self.degree_dict:
                    degree_word[index] = self.degree_dict[word]
            return sen_word, not_word, degree_word


        def get2score_position(self, words):
            sen_word, not_word, degree_word =  self.classify_words(words)   # 是字典

            score = 0
            start = 0
            # 存所有情感词、否定词、程度副词的位置(索引、序号)的列表
            sen_locs = sen_word.keys()
            not_locs = not_word.keys()
            degree_locs = degree_word.keys()
            senloc = -1
            # 遍历句子中所有的单词words，i为单词的绝对位置
            for i in range(0, len(words)):
                if i in sen_locs:
                    W = 1  # 情感词间权重重置
                    not_locs_index = 0
                    degree_locs_index = 0

                    # senloc为情感词位置列表的序号,之前的sen_locs是情感词再分词后列表中的位置序号
                    senloc += 1
                    #score += W * float(sen_word[i])
                    if (senloc==0): # 第一个情感词,前面是否有否定词，程度词
                        start = 0
                    elif senloc < len(sen_locs):  # 和前面一个情感词之间，是否有否定词,程度词
                        # j为绝对位置
                        start = previous_sen_locs

                    for j in range(start,i): # 词间的相对位置
                        # 如果有否定词
                        if j in not_locs:
                            W *= -1
                            not_locs_index=j
                        # 如果有程度副词
                        elif j in degree_locs:
                            W *= degree_word[j]
                            degree_locs_index=j

                        # 判断否定词和程度词的位置：1）否定词在前，程度词减半(加上正值)；不是很   2）否定词在后，程度增强（不变），很不是
                    if ((not_locs_index>0) and (degree_locs_index>0 )):
                        if (not_locs_index < degree_locs_index ):
                            degree_reduce = (float(degree_word[degree_locs_index]/2))
                            W +=degree_reduce
                            #print (W)
                    score += W * float(sen_word[i])  # 直接添加该情感词分数
                    #print(score)
                    previous_sen_locs = i
            return score

        #感觉get2score用处不是很大
        def get2score(self, text):
            word_list = self.findword(text)  ##查找文本中包含哪些正负情感词，然后分别分别累计它们的数值
            pos_score = 0
            pos_word = []
            neg_score = 0
            neg_word=[]
            for word in word_list:
                if (word.value>0):
                    pos_score = pos_score + word.value
                    pos_word.append(word.word)
                else:
                    neg_score = neg_score+word.value
                    neg_word.append(word.word)
            print ("pos_score=%d; neg_score=%d" %(pos_score, neg_score))
            #print('pos_word',pos_word)
            #print('neg_word',neg_word)

        def getscore(self, text):
            word_list = self.findword(text)  ##查找文本中包含哪些情感词
            # 增加程度副词+否定词
            not_w = 1
            not_word = []
            for notword in self.__class__.NOT_DICT:  # 否定词
                if notword in text:
                    not_w = not_w * -1
                    not_word.append(notword)
            degree_word = []
            for degreeword in self.degree_dict.keys():
                if degreeword in text:
                    degree = self.degree_dict[degreeword]
                    #polar = polar + degree if polar > 0 else polar - degree
                    degree_word.append(degreeword)
            # 7大类找对应感情大类的词语，分别统计分数= 词极性*词权重
            result = []
            for key in self.score_class.keys(): #区分7大类
                score = 0
                score_words = []
                for word in word_list:
                    
                    if (key == word.class_value):
                        score = score + word.value
                        score_words.append(word.word)
                if score > 0:
                    score = score + degree
                elif score<0:
                    score = score - degree  # 看分数>0，程度更强； 分数<0,程度减弱？
                score = score * not_w

                x = '{}_score={}; word={}; nor_word={}; degree_word={};'.format(key, score, score_words,not_word, degree_word)
                print (x)
                result.append(x)
                #key + '_score=%d; word=%s; nor_word=%s; degree_word=%s;'% (score, score_words,not_word, degree_word))
            return result

if __name__ == '__main__':
    sentiment_dict_path = "sentiment_words_chinese.tsv" 
    degree_dict_path = "degree_dict.txt"
    stop_dict_path = "stop_words.txt"

    #文件读取
    f = open('庆余年220.csv',encoding='utf8')
    data = pd.read_csv(f)

    #文件写入
    c = open("Result.csv", "a+", newline='', encoding='gb18030')
    writer = csv.writer(c)
    writer.writerow(["no","review","score"])

    #分句功能 否定词程度词位置判断
    score = Score(sentiment_dict_path, degree_dict_path, stop_dict_path )

    n = 1
    for temp in data['review']:
        tlist = []
        words = [x for x in jieba.cut(temp)] #分词
        #print(words)     
        words_ = score.remove_stopword(words)
        print(words_)
        
        #分词->情感词间是否有否定词/程度词+前后顺序->分数累加
        result = score.get2score_position(words_)  
        print(result)
        
        tlist.append(str(n))
        tlist.append(words)
        tlist.append(str(result))
        writer.writerow(tlist)
        n = n + 1

        #句子-> 整句判断否定词/程度词 -> 分正负词
        #score.get2score(temp) 
        #score.getscore(text)
    c.close()

