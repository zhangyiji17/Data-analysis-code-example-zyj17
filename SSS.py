from datetime import datetime
import sys
import numpy as np
from PIL import Image
import boto3
import botocore
import ssl
import urllib3
from urllib.parse import urlparse
import cv2
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Struct import *
from NetSDK.SDK_Enum import *
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect, CB_FUNCTYPE
import time
from datetime import datetime
import keyboard
import io
import pymysql
import threading
import ctypes

video_stream_path = f"rtsp://admin:sunward12345@10.1.28.184:80/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(video_stream_path)  # 连接摄像头
image = cap.read()[1]
print(type(image))
im1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image0 = im1.astype(np.uint8)
image1 = Image.fromarray(image0)
with io.BytesIO() as output:
    image1.save(output, format='JPEG')
    image_upload = output.getvalue()

timenow = str(datetime.now())
pic_name = timenow[:4] + timenow[5:7] + timenow[8:10] + '_' + timenow[11:13] + timenow[14:16] + timenow[17:19]
# 上传图片到桶
######################################################################################################
urllib3.disable_warnings()  # 禁用安全请求警告
ssl._create_default_https_context = ssl._create_unverified_context

bucket = 'machinevisionbucket01'  # 存储桶的名称
AKI = '0UVJFIWO9Q7Z25FFKBG1'  # 访问秘钥
SAK = 'viN0JM9bv0IDfAgeXzWPspTznACAbQYHXKauB5PK'  # 安全秘钥
url = 'https://xsky.sunward.cn:443'  # 地址
print(1)
s3 = boto3.client('s3',
                  endpoint_url=url,
                  aws_access_key_id=AKI,
                  aws_secret_access_key=SAK,
                  verify=False  # 不验证链接
                  )
path = 'huoqing/' + pic_name + '.jpg'
print(2)
s3.put_object(Bucket=bucket, Key=path, Body=image_upload, ContentType='image/jpeg')
print(3)
url_parsed = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': path})

clean_url = (urlparse(url_parsed).scheme + '://smartfactory.sunwardcloud.cn/xsky' + urlparse(url_parsed).path)
response = s3.head_object(Bucket=bucket, Key=path)
size = round(response['ContentLength'] / 1024, 1)

s3.close()
cap.release()
print('截图完成')