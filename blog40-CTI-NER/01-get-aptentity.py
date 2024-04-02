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
filename = []
for name in names:
    filename.append(name.strip())
print(filename)

#链接
urls = html_etree.xpath('//*[@class="table table-bordered table-alternate mt-2"]/tbody/tr/td[2]/a/@href')
print(urls)
print(len(urls), urls[0])
print("\n")
