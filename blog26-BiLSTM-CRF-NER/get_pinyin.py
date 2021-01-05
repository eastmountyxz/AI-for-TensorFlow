from cnradical import Radical, RunOption

radical = Radical(RunOption.Radical)
pinyin = Radical(RunOption.Pinyin)

text = '你好，今天早上吃饭了吗？Eastmount'
radical_out = [radical.trans_ch(ele) for ele in text]
pinyin_out = [pinyin.trans_ch(ele) for ele in text]
print(radical_out)
print(pinyin_out)

radical_out = radical.trans_str(text)
pinyin_out = pinyin.trans_str(text)
print(radical_out)
print(pinyin_out)
