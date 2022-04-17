"""
[kakao vision api]
사람 연령/성별 분석을 위한 api 호출
kakao vision api에 가입 한 뒤, API_URL과 MYAPP_KEY를 지정해줘야 정상적으로 구동 됨
"""

import cv2
import sys
import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = 'https://dapi.kakao.com/v2/vision/face/detect'
MYAPP_KEY = 'e0cffebf1a36764eefc9c491594f5a3d'


def detect_face_image(image):
    """ 얼굴 검출 및 성별/연령 분석 """
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    try:
        jpeg_image = cv2.imencode(".jpg", image)[1]
        #cv2.imwrite("img.jpg",image)
        data = jpeg_image.tobytes()
        #files = { 'image' : open(filename, 'rb')}
        resp = requests.post(API_URL, headers=headers, files={"image": data})
        #resp = requests.post(API_URL, headers=headers, files=files)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)




