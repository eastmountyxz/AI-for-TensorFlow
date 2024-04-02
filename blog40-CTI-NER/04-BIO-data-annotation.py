#encoding:utf-8
import re
import os
import csv

#-----------------------------------------定义实体类型-------------------------------------
#APT攻击组织
aptName = ['admin@338', 'Ajax Security Team', 'APT-C-36', 'APT1', 'APT12', 'APT16', 'APT17', 'APT18', 'APT19', 'APT28', 'APT29', 'APT3', 'APT30', 'APT32',
           'APT33', 'APT37', 'APT38', 'APT39', 'APT41', 'Axiom', 'BlackOasis', 'BlackTech', 'Blue Mockingbird', 'Bouncing Golf', 'BRONZE BUTLER',
           'Carbanak', 'Chimera', 'Cleaver', 'Cobalt Group', 'CopyKittens', 'Dark Caracal', 'Darkhotel', 'DarkHydrus', 'DarkVishnya', 'Deep Panda',
           'Dragonfly', 'Dragonfly 2.0', 'DragonOK', 'Dust Storm', 'Elderwood', 'Equation', 'Evilnum', 'FIN10', 'FIN4', 'FIN5', 'FIN6', 'FIN7', 'FIN8',
           'Fox Kitten', 'Frankenstein', 'GALLIUM', 'Gallmaker', 'Gamaredon Group', 'GCMAN', 'GOLD SOUTHFIELD', 'Gorgon Group', 'Group5', 'HAFNIUM',
           'Higaisa', 'Honeybee', 'Inception', 'Indrik Spider', 'Ke3chang', 'Kimsuky', 'Lazarus Group', 'Leafminer', 'Leviathan', 'Lotus Blossom',
           'Machete', 'Magic Hound', 'menuPass', 'Moafee', 'Mofang', 'Molerats', 'MuddyWater', 'Mustang Panda', 'Naikon', 'NEODYMIUM', 'Night Dragon',
           'OilRig', 'Operation Wocao', 'Orangeworm', 'Patchwork', 'PittyTiger', 'PLATINUM', 'Poseidon Group', 'PROMETHIUM', 'Putter Panda', 'Rancor',
           'Rocke', 'RTM', 'Sandworm Team', 'Scarlet Mimic', 'Sharpshooter', 'Sidewinder', 'Silence', 'Silent Librarian', 'SilverTerrier', 'Sowbug', 'Stealth Falcon',
           'Stolen Pencil', 'Strider', 'Suckfly', 'TA459', 'TA505', 'TA551', 'Taidoor', 'TEMP.Veles', 'The White Company', 'Threat Group-1314', 'Threat Group-3390',
           'Thrip', 'Tropic Trooper', 'Turla', 'Volatile Cedar', 'Whitefly', 'Windigo', 'Windshift', 'Winnti Group', 'WIRTE', 'Wizard Spider', 'ZIRCONIUM',
           'UNC2452', 'NOBELIUM', 'StellarParticle']

#特殊名称的攻击漏洞
cveName = ['CVE-2009-3129', 'CVE-2012-0158', 'CVE-2009-4324' 'CVE-2009-0927', 'CVE-2011-0609', 'CVE-2011-0611', 'CVE-2012-0158',
           'CVE-2017-0262', 'CVE-2015-4902', 'CVE-2015-1701', 'CVE-2014-4076', 'CVE-2015-2387', 'CVE-2015-1701', 'CVE-2017-0263']

#区域位置
locationName = ['China-based', 'China', 'North', 'Korea', 'Russia', 'South', 'Asia', 'US', 'U.S.', 'UK', 'U.K.', 'Iran', 'Iranian', 'America', 'Colombian',
                'Chinese', "People’s",  'Liberation', 'Army', 'PLA', 'General', 'Staff', "Department’s", 'GSD', 'MUCD', 'Unit', '61398', 'Chinese-based',
                "Russia's", "General", "Staff", "Main", "Intelligence", "Directorate", "GRU", "GTsSS", "unit", "26165", '74455', 'Georgian', 'SVR',
                'Europe', 'Asia', 'Hong Kong', 'Vietnam', 'Cambodia', 'Thailand', 'Germany', 'Spain', 'Finland', 'Israel', 'India', 'Italy', 'South Asia',
                'Korea', 'Kuwait', 'Lebanon', 'Malaysia', 'United', 'Kingdom', 'Netherlands', 'Southeast', 'Asia', 'Pakistan', 'Canada', 'Bangladesh',
                'Ukraine', 'Austria', 'France', 'Korea']

#攻击行业
industryName = ['financial', 'economic', 'trade', 'policy', 'defense', 'industrial', 'espionage', 'government', 'institutions', 'institution', 'petroleum',
                'industry', 'manufacturing', 'corporations', 'media', 'outlets', 'high-tech', 'companies', 'governments', 'medical', 'defense', 'finance',
                'energy', 'pharmaceutical', 'telecommunications', 'high', 'tech', 'education', 'investment', 'firms', 'organizations', 'research', 'institutes',
                ]

#攻击方法
methodName = ['RATs', 'RAT', 'SQL', 'injection', 'spearphishing', 'spear', 'phishing', 'backdoors', 'vulnerabilities', 'vulnerability', 'commands', 'command',
              'anti-censorship', 'keystrokes', 'VBScript', 'malicious', 'document', 'scheduled', 'tasks', 'C2', 'C&C', 'communications', 'batch', 'script',
              'shell', 'scripting', 'social', 'engineering', 'privilege', 'escalation', 'credential', 'dumping', 'control', 'obfuscates', 'obfuscate', 'payload', 'upload',
              'payloads', 'encode', 'decrypts', 'attachments', 'attachment', 'inject', 'collect', 'large-scale', 'scans', 'persistence', 'brute-force/password-spray',
              'password-spraying', 'backdoor', 'bypass', 'hijacking', 'escalate', 'privileges', 'lateral', 'movement', 'Vulnerability', 'timestomping',
              'keylogging', 'DDoS', 'bootkit', 'UPX' ]

#利用软件
softwareName = ['Microsoft', 'Word', 'Office', 'Firefox', 'Google', 'RAR', 'WinRAR', 'zip', 'GETMAIL', 'MAPIGET', 'Outlook', 'Exchange', "Adobe's", 'Adobe',
                'Acrobat', 'Reader', 'RDP', 'PDFs', 'PDF', 'RTF', 'XLSM', 'USB', 'SharePoint', 'Forfiles', 'Delphi', 'COM', 'Excel', 'NetBIOS',
                'Tor', 'Defender', 'Scanner', 'Gmail', 'Yahoo', 'Mail', '7-Zip', 'Twitter', 'gMSA', 'Azure', 'Exchange', 'OWA', 'SMB', 'Netbios',
                'WinRM']

#操作系统
osName = ['Windows', 'windows', 'Mac', 'Linux', 'Android', 'android', 'linux', 'mac', 'unix', 'Unix']

#计算并输出相关的内容
saveCVE = cveName
saveAPT = aptName
saveLocation = locationName
saveIndustry = industryName
saveMethod = methodName
saveSoftware = softwareName
saveOS = osName

#------------------------------------------------------------------------
#获取文件路径及名称
def get_filepath(path):
    entities = {}              #字段实体类别
    files = os.listdir(path)   #遍历路径
    return files

#-----------------------------------------------------------------------
#获取文件内容
def get_content(filename):
    content = []
    with open(filename, "r", encoding="utf8") as f:
        for line in f.readlines():
            content.append(line.strip())
    return content
            
#---------------------------------------------------------------------
#空格分隔获取英文单词
def data_annotation(text):
    n = 0
    nums = []
    while n<len(text):
        word = text[n].strip()
        if word == "":   #换行 startswith
            n += 1
            nums.append("")
            continue
        
        #APT攻击组织
        if word in aptName:
            nums.append("B-AG")
        #攻击漏洞
        elif "CVE-" in word or 'MS-' in word:
            nums.append("B-AV")
            print("CVE漏洞:", word)
            if word not in saveCVE:
                saveCVE.append(word)
        #区域位置
        elif word in locationName:
            nums.append("B-RL")
        #攻击行业
        elif word in industryName:
            nums.append("B-AI")
        #攻击手法
        elif word in methodName:
            nums.append("B-AM")
        #利用软件
        elif word in softwareName:
            nums.append("B-SI")
        #操作系统
        elif word in osName:
            nums.append("B-OS")
       
        #特殊情况-APT组织
        #Ajax Security Team、Deep Panda、Sandworm Team、Cozy Bear、The Dukes、Dark Halo
        elif ((word in "Ajax Security Team") and (text[n+1].strip() in "Ajax Security Team") and word!="a" and word!="it") or \
              ((word in "Ajax Security Team") and (text[n-1].strip() in "Ajax Security Team") and word!="a" and word!="it") or \
              ((word=="Deep") and (text[n+1].strip()=="Panda")) or \
              ((word=="Panda") and (text[n-1].strip()=="Deep")) or \
              ((word=="Sandworm") and (text[n+1].strip()=="Team")) or \
              ((word=="Team") and (text[n-1].strip()=="Sandworm")) or \
              ((word=="Cozy") and (text[n+1].strip()=="Bear")) or \
              ((word=="Bear") and (text[n-1].strip()=="Cozy")) or \
              ((word=="The") and (text[n+1].strip()=="Dukes")) or \
              ((word=="Dukes") and (text[n-1].strip()=="The")) or \
              ((word=="Dark") and (text[n+1].strip()=="Halo")) or \
              ((word=="Halo") and (text[n-1].strip()=="Dark")):
            nums.append("B-AG")
            if "Deep Panda" not in saveAPT:
                saveAPT.append("Deep Panda")
            if "Sandworm Team" not in saveAPT:
                saveAPT.append("Sandworm Team")
            if "Cozy Bear" not in saveAPT:
                saveAPT.append("Cozy Bear")
            if "The Dukes" not in saveAPT:
                saveAPT.append("The Dukes")
            if "Dark Halo" not in saveAPT:
                saveAPT.append("Dark Halo")     
         
        #特殊情况-攻击行业
        elif ((word=="legal") and (text[n+1].strip()=="services")) or \
              ((word=="services") and (text[n-1].strip()=="legal")):
            nums.append("B-AI")
            if "legal services" not in saveIndustry:
                saveIndustry.append("legal services")
                
        #特殊情况-攻击方法
        #watering hole attack、bypass application control、take screenshots
        elif ((word in "watering hole attack") and (text[n+1].strip() in "watering hole attack") and word!="a" and text[n+1].strip()!="a") or \
              ((word in "watering hole attack") and (text[n-1].strip() in "watering hole attack") and word!="a" and text[n+1].strip()!="a") or \
              ((word in "bypass application control") and (text[n+1].strip() in "bypass application control") and word!="a" and text[n+1].strip()!="a") or \
              ((word in "bypass application control") and (text[n-1].strip() in "bypass application control") and word!="a" and text[n-1].strip()!="a") or \
              ((word=="take") and (text[n+1].strip()=="screenshots")) or \
              ((word=="screenshots") and (text[n-1].strip()=="take")):
            nums.append("B-AM")
            if "watering hole attack" not in saveMethod:
                saveMethod.append("watering hole attack")
            if "bypass application control" not in saveMethod:
                saveMethod.append("bypass application control")
            if "take screenshots" not in saveMethod:
                saveMethod.append("take screenshots")
                
        #特殊情况-利用软件
        #MAC address、IP address、Port 22、Delivery Service、McAfee Email Protection
        elif ((word=="legal") and (text[n+1].strip()=="services")) or \
              ((word=="services") and (text[n-1].strip()=="legal")) or \
              ((word=="MAC") and (text[n+1].strip()=="address")) or \
              ((word=="address") and (text[n-1].strip()=="MAC")) or \
              ((word=="IP") and (text[n+1].strip()=="address")) or \
              ((word=="address") and (text[n-1].strip()=="IP")) or \
              ((word=="Port") and (text[n+1].strip()=="22")) or \
              ((word=="22") and (text[n-1].strip()=="Port")) or \
              ((word=="Delivery") and (text[n+1].strip()=="Service")) or \
              ((word=="Service") and (text[n-1].strip()=="Delivery")) or \
              ((word in "McAfee Email Protection") and (text[n+1].strip() in "McAfee Email Protection")) or \
              ((word in "McAfee Email Protection") and (text[n-1].strip() in "McAfee Email Protection")):
            nums.append("B-SI")
            if "MAC address" not in saveSoftware:
                saveSoftware.append("MAC address")
            if "IP address" not in saveSoftware:
                saveSoftware.append("IP address")
            if "Port 22" not in saveSoftware:
                saveSoftware.append("Port 22")
            if "Delivery Service" not in saveSoftware:
                saveSoftware.append("Delivery Service")
            if "McAfee Email Protection" not in saveSoftware:
                saveSoftware.append("McAfee Email Protection")
   
        #特殊情况-区域位置
        #Russia's Foreign Intelligence Service、the Middle East
        elif ((word in "Russia's Foreign Intelligence Service") and (text[n+1].strip() in "Russia's Foreign Intelligence Service")) or \
             ((word in "Russia's Foreign Intelligence Service") and (text[n-1].strip() in "Russia's Foreign Intelligence Service")) or \
             ((word in "the Middle East") and (text[n+1].strip() in "the Middle East")) or \
             ((word in "the Middle East") and (text[n-1].strip() in "the Middle East")) :
            nums.append("B-RL")
            if "Russia's Foreign Intelligence Service" not in saveLocation:
                saveLocation.append("Russia's Foreign Intelligence Service")
            if "the Middle East" not in saveLocation:
                saveLocation.append("the Middle East")
            
        else:
            nums.append("O")
        n += 1
    return nums
    
#-----------------------------------------------------------------------
#主函数
if __name__ == '__main__':
    path = "Mitre-Split-Word"
    savepath = "Mitre-Split-Word-BIO"
    filenames = get_filepath(path)
    print(filenames)
    print("\n")

    #遍历文件内容
    k = 0
    while k<len(filenames):
        filename = path + "//" + filenames[k]
        print("-------------------------")
        print(filename)
        content = get_content(filename)

        #分割句子
        nums = data_annotation(content)
        #print(nums)
        print(len(content),len(nums))

        #数据存储
        filename = filenames[k].replace(".txt", ".csv")
        savename = savepath + "//" + filename
        f = open(savename, "w", encoding="utf8", newline='')
        fwrite = csv.writer(f)
        fwrite.writerow(['word','label'])
        n = 0
        while n<len(content):
            fwrite.writerow([content[n],nums[n]])
            n += 1
        f.close()
        print("-------------------------\n\n")
        
        #if k>=28:
        #    break
        k += 1

    #-------------------------------------------------------------------------------------------------
    #输出存储的漏洞结果
    saveCVE.remove("CVE-2009-4324CVE-2009-0927")
    saveCVE.sort()
    print(saveCVE)
    print("CVE漏洞:", len(saveCVE))

    saveAPT.sort()
    print(saveAPT)
    print("APT组织:", len(saveAPT))

    saveLocation.sort()
    print(saveLocation)
    print("区域位置:", len(saveLocation))

    saveIndustry.sort()
    print(saveIndustry)
    print("攻击行业:", len(saveIndustry))

    saveSoftware.sort()
    print(saveSoftware)
    print("利用软件:", len(saveSoftware))

    saveMethod.sort()
    print(saveMethod)
    print("攻击手法:", len(saveMethod))

    saveOS.sort()
    print(saveOS)
    print("操作系统:", len(saveOS))

