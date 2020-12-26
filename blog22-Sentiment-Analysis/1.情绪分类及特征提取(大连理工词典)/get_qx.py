# coding: utf-8
import pandas as pd
import jieba
import time

#-------------------------------------获取数据集---------------------------------
f = open('庆余年220.csv',encoding='utf8')
weibo_df = pd.read_csv(f)
print(weibo_df.head())

#-------------------------------------情感词典读取-------------------------------
#注意：
#1.词典中怒的标记(NA)识别不出被当作空值,情感分类列中的NA都给替换成NAU
#2.大连理工词典中有情感分类的辅助标注(有NA),故把情感分类列改好再替换原词典中

# 扩展前的词典
df = pd.read_excel('大连理工大学中文情感词汇本体NAU.xlsx')
print(df.head(10))

df = df[['词语', '词性种类', '词义数', '词义序号', '情感分类', '强度', '极性']]
df.head()

#-------------------------------------七种情绪的运用-------------------------------
Happy = []
Good = []
Surprise = []
Anger = []
Sad = []
Fear = []
Disgust = []

#df.iterrows()功能是迭代遍历每一行
for idx, row in df.iterrows():
    if row['情感分类'] in ['PA', 'PE']:
        Happy.append(row['词语'])
    if row['情感分类'] in ['PD', 'PH', 'PG', 'PB', 'PK']:
        Good.append(row['词语']) 
    if row['情感分类'] in ['PC']:
        Surprise.append(row['词语'])       
    if row['情感分类'] in ['NB', 'NJ', 'NH', 'PF']:
        Sad.append(row['词语'])
    if row['情感分类'] in ['NI', 'NC', 'NG']:
        Fear.append(row['词语'])
    if row['情感分类'] in ['NE', 'ND', 'NN', 'NK', 'NL']:
        Disgust.append(row['词语'])
    if row['情感分类'] in ['NAU']:     #修改: 原NA算出来没结果
        Anger.append(row['词语'])  

#正负计算不是很准 自己可以制定规则       
Positive = Happy + Good + Surprise
Negative = Anger + Sad + Fear + Disgust
print('情绪词语列表整理完成')  
print(Anger)

#---------------------------------------中文分词---------------------------------

#添加自定义词典和停用词
#jieba.load_userdict("user_dict.txt")
stop_list = pd.read_csv('stop_words.txt',
                        engine='python',
                        encoding='utf-8',
                        delimiter="\n",
                        names=['t'])

#获取重命名t列的值
stop_list = stop_list['t'].tolist()

def txt_cut(juzi):
    return [w for w in jieba.lcut(juzi) if w not in stop_list]     #可增加len(w)>1

#---------------------------------------情感计算---------------------------------
def emotion_caculate(text):
    positive = 0
    negative = 0
    
    anger = 0
    disgust = 0
    fear = 0
    sad = 0
    surprise = 0
    good = 0
    happy = 0

    anger_list = []
    disgust_list = []
    fear_list = []
    sad_list = []
    surprise_list = []
    good_list = []
    happy_list = []
    
    wordlist = txt_cut(text)
    #wordlist = jieba.lcut(text)
    wordset = set(wordlist)
    wordfreq = []
    for word in wordset:
        freq = wordlist.count(word)
        if word in Positive:
            positive+=freq
        if word in Negative:
            negative+=freq
        if word in Anger:
            anger+=freq
            anger_list.append(word)
        if word in Disgust:
            disgust+=freq
            disgust_list.append(word)
        if word in Fear:
            fear+=freq
            fear_list.append(word)
        if word in Sad:
            sad+=freq
            sad_list.append(word)
        if word in Surprise:
            surprise+=freq
            surprise_list.append(word)
        if word in Good:
            good+=freq
            good_list.append(word)
        if word in Happy:
            happy+=freq
            happy_list.append(word)
            
    emotion_info = {
        'length':len(wordlist),
        'positive': positive,
        'negative': negative,
        'anger': anger,
        'disgust': disgust,
        'fear':fear,
        'good':good,
        'sadness':sad,
        'surprise':surprise,
        'happy':happy,
        
    }

    indexs = ['length', 'positive', 'negative', 'anger', 'disgust','fear','sadness','surprise', 'good', 'happy']
    #return pd.Series(emotion_info, index=indexs), anger_list, disgust_list, fear_list, sad_list, surprise_list, good_list, happy_list
    return pd.Series(emotion_info, index=indexs)

#测试 (res, anger_list, disgust_list, fear_list, sad_list, surprise_list, good_list, happy_list)
text = """
原著的确更吸引编剧读下去，所以跟《诛仙》系列明显感觉到编剧只看过故事大纲比，这个剧的编剧完整阅读过小说。
配乐活泼俏皮，除了强硬穿越的台词轻微尴尬，最应该尴尬的感情戏反而入戏，故意模糊了陈萍萍的太监身份、太子跟长公主的暧昧关系，
整体观影感受极好，很期待第二季拍大东山之役。玩弄人心的阴谋阳谋都不狗血，架空的设定能摆脱历史背景，服装道具能有更自由的发挥空间，
特别喜欢庆帝的闺房。以后还是少看国产剧，太长了，还是精短美剧更适合休闲，追这个太累。王启年真是太可爱了。
"""
#res, anger, disgust, fear, sad, surprise, good, happy = emotion_caculate(text)
res = emotion_caculate(text)
print(res)

#---------------------------------------情感计算---------------------------------
start = time.time()   
emotion_df = weibo_df['review'].apply(emotion_caculate)
end = time.time()
print(end-start)
print(emotion_df.head())

#输出结果
output_df = pd.concat([weibo_df, emotion_df], axis=1)
output_df.to_csv('庆余年220_emotion.csv',encoding='utf_8_sig', index=False)
print(output_df.head())

#显示fear、negative数据集
fear_content = output_df.sort_values(by='fear',ascending=False)
print(fear_content)
print(fear_content.iloc[0:5]['review'])

negative_content = output_df.sort_values(by='negative',ascending=False)
print(negative_content)
print(negative_content.iloc[0:5]['review'])
