# coding=utf-8
# =============================================
# @Time      : 2023-01-04 14:57
# @Author    : DongWei1998
# @FileName  : flasktest.py.py
# @Software  : PyCharm
# =============================================


import time
import requests
import json
def ner_post():
    url = f"http://127.0.0.1:8888/app/v1/ner"
    demo_text = {
        "text":"作任何修改或补充，须由双方以书面做出方为有效。12.5本协议一式【叁】份，甲方执【壹】份，乙方执【贰】份，自双方签字盖章之日起生效。第十三条协议附件附件：【/】（以下无正文）甲方：【山东家家悦超市有限公司】（盖章）签字："
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers,)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))

if __name__ == '__main__':
    ner_post()