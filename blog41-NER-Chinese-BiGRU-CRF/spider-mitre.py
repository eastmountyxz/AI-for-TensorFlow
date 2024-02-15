#encoding:utf-8
#By:Eastmount CSDN
import re
import requests
from lxml import etree
from bs4 import BeautifulSoup
import urllib.request

#-------------------------------------------------------------------------------------------
#获取APT组织名称及链接

#设置浏览器代理,它是一个字典
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
}
url = 'https://attack.mitre.org/groups/'

#向服务器发出请求
r = requests.get(url = url, headers = headers).text

#解析DOM树结构
html_etree = etree.HTML(r)
names = html_etree.xpath('//*[@class="table table-bordered table-alternate mt-2"]/tbody/tr/td[2]/a/text()')
print (names)
print(len(names),names[0])

#链接
urls = html_etree.xpath('//*[@class="table table-bordered table-alternate mt-2"]/tbody/tr/td[2]/a/@href')
print(urls)
print(len(urls), urls[0])
print("\n")


#-------------------------------------------------------------------------------------------
#获取详细信息
k = 0
while k<len(names):
    filename = str(names[k]).strip() + ".txt"
    url = "https://attack.mitre.org" + urls[k]
    print(url)

    #获取正文信息
    page = urllib.request.Request(url, headers=headers)
    page = urllib.request.urlopen(page)
    contents = page.read()
    soup = BeautifulSoup(contents, "html.parser")

    #获取正文摘要信息
    content = ""
    for tag in soup.find_all(attrs={"class":"description-body"}):
        #contents = tag.find("p").get_text()
        contents = tag.find_all("p")
        for con in contents:
            content += con.get_text().strip() + "###\n"  #标记句子结束(第二部分分句用)
    #print(content)

    #获取表格中的技术信息
    for tag in soup.find_all(attrs={"class":"table techniques-used table-bordered mt-2"}):
        contents = tag.find("tbody").find_all("tr")
        for con in contents:
            value = con.find("p").get_text()           #存在4列或5列 故获取p值
            #print(value)
            content += value.strip() + "###\n"         #标记句子结束(第二部分分句用)

    #删除内容中的参考文献括号 [n]
    result = re.sub(u"\\[.*?]", "", content)
    print(result)

    #文件写入
    filename = "Mitre//" + filename
    print(filename)
    f = open(filename, "w", encoding="utf-8")
    f.write(result)
    f.close()    
    k += 1
