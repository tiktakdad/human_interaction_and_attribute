"""
[소켓 클라이언트]
키오스크 WAS 서버와 결과값 전송을 위한 소켓통신 모듈
"""
import datetime
import json
import threading
import socket
from collections import OrderedDict

import cv2
import time

class SocketClient(threading.Thread):
    """ 소켓 클라이언트 클래스 """
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.data = None
        self.exit_flag = False
        self.lock = threading.Lock()
        self.connect()

    def insert_data(self, json_array):
        """ 비동기식 json 데이터 추가 """
        self.lock.acquire()
        try:
            has_data = False
            datas = {}
            datas['results'] = {}
            datas['results']['sys_time'] = str(datetime.datetime.now())
            for json_data in json_array:
                if json_data is not None:
                    key = list(json_data.keys())[0]
                    datas['results'][key] = json_data[key]
                    has_data = True
            self.data = json.dumps(datas)
            if has_data is False:
                self.data = None
        finally:
            self.lock.release()


    def connect(self):
        """ 서버 접속 """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #sock.connect((FLAGS.ip_address, FLAGS.port))
            self.sock.connect((self.host, self.port))
            print("(Check the new thread) " + self.host + ":" + str(self.port))
            return True
        except:
            print("Could not make a connection to the server")
            return False

    def release(self):
        self.sock.close()
        self.exit_flag = True


    def run(self) -> None:
        """ 클라이언트 데이터 전송 쓰레드 """
        while True:
            time.sleep(0.0001)
            if self.exit_flag is True:
                break
            if self.data is not None:
                '''
                meta_result = []
                datas = OrderedDict()
                datas['roi'] = [[0, 0], [1, 1], [2, 2], [3, 3]]
                datas['label'] = '{}'.format('person')
                meta_result.append(datas)
                result = json.dumps(meta_result)
                '''

                self.sock.sendall(len(str.encode(self.data)).to_bytes(4, byteorder="big"))
                self.sock.sendall(str.encode(self.data))
                #print('send :', self.data)
                self.lock.acquire()
                try:
                    self.data = None
                finally:
                    # Lock을 해제해서 다른 Thread도 사용할 수 있도록 만든다.
                    self.lock.release()

