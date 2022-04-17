"""
[얼굴 속성분석]
468개의 얼굴 랜드마크 데이터를 이용하여 얼굴 윤곽을 검출하고, 연령/성별 분석을 함
별도의 쓰레드로 돌아가며 attribute_interval에 설정 된 시간(초)마다 연령/성별 얼굴 분석을 실행하고 평균 값을 취함
"""
import cv2
import numpy as np
import threading
from kakao_api import detect_face_image
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION


class FaceAttributeAnalyzer:
    """
    얼굴 속성분석 클래스
    """
    def __init__(self, attribute_interval, connections):
        self.face_attribute = ThreadFaceAttributeVariable()
        self.id = None
        self.interval = attribute_interval
        self.point_list = []
        self.coordinates = None
        self.image_width = None
        self.image_height = None
        self.connections = connections

    def analyze_face(self, image, coordinates, id):
        """
           얼굴 속성분석 실행
           face id가 유지되는 동안 연령/성별 평균 값을 유지 함
        """
        if self.id is not id:
            self.reset()

        rect = self.get_rect_from_landmarks(coordinates)
        self.coordinates = coordinates
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.point_list = np.array(list(coordinates.values()))

        if self.id is None:
            self.id = id
            print('set id:', self.id)

        if self.analyzer is None or self.analyzer.is_alive() is False:
            if self.face_attribute.counter < 10:
                self.analyzer = threading.Timer(interval=self.interval, function=self.get_face_data,
                                                args=(image, rect, self.face_attribute, False))
                # print('self.analyzer:', self.analyzer)
                self.analyzer.start()


    def reset(self):
        self.face_attribute.reset()
        self.analyzer = None
        self.id = None

    def get_gender_age(self):
        """ kakao vision api를 통해 성별/연령 데이터 취득 """
        return self.face_attribute.gender, self.face_attribute.age




    def get_rect_from_landmarks(self, coordinates):
        """ 랜드마크에서 바운딩 박스를 얻음 """
        point_list = np.array(list(coordinates.values()))
        x_min, y_min = point_list.min(axis=0)
        x_max, y_max = point_list.max(axis=0)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def get_connections_from_landmarks(self, coordinates, connections):
        """ 얼굴 전체 랜드마크를 입술, 왼쪽눈, 왼쪽눈썹, 오른쪽눈, 오른쪽눈썹, 얼굴윤곽 순으로 분리 """
        lips = []
        left_eye = []
        left_eyebrow = []
        right_eye = []
        right_eyebrow = []
        face_oval = []
        if connections:
            num_landmarks = len(coordinates)
            # Draws the connections if the start and end landmarks are both visible.
            for i, parts in enumerate(connections):
                for connection in parts:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                        return lips, left_eye, left_eyebrow, right_eye, right_eyebrow, face_oval
                    if start_idx in coordinates and end_idx in coordinates:
                        if i == 0:  # lip
                            lips.append([coordinates[start_idx], coordinates[end_idx]])
                            # cv2.line(image, coordinates[start_idx],coordinates[end_idx], (255,0,0), 1)
                        elif i == 1:  # left_eye
                            left_eye.append([coordinates[start_idx], coordinates[end_idx]])
                        elif i == 2:  # left_eyebrow
                            left_eyebrow.append([coordinates[start_idx], coordinates[end_idx]])
                        elif i == 3:  # right_eye
                            right_eye.append([coordinates[start_idx], coordinates[end_idx]])
                        elif i == 4:  # right_eyebrow
                            right_eyebrow.append([coordinates[start_idx], coordinates[end_idx]])
                        elif i == 5:  # face_oval
                            face_oval.append([coordinates[start_idx], coordinates[end_idx]])
        return lips, left_eye, left_eyebrow, right_eye, right_eyebrow, face_oval

    def draw_parts_number(self, img, color):
        """ 얼굴 중요 포인트들을 그림 """
        # Nose tip : 4
        # Left Mouth corner : 61
        # Left eye left corner :  130
        # Chin : 152
        # Right mouth corner : 291
        # Right eye right corner : 359
        important_points = [4, 61, 130, 152, 291, 359]
        #important_points = []
        for idx in self.coordinates:
            cv2.circle(img, self.coordinates[idx], 1, color)
            cv2.putText(img, str(idx), self.coordinates[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            color, thickness=1, lineType=cv2.LINE_AA)

        for imp in important_points:
            imp_coord = self.coordinates.get(imp)

            print(imp_coord)
            # imp_landmark = landmarks_list.landmark[imp]
            if imp_coord is not None:
                cv2.circle(img, self.coordinates[imp], 1, (0, 0, 255))
                cv2.putText(img, str(imp), self.coordinates[imp], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255),
                            thickness=1, lineType=cv2.LINE_AA)


    def draw_parts(self, img, tessellation=True):
        """ 얼굴의 모든 파트들을 그림 """
        if tessellation:
            self.draw_part(img, self.coordinates, FACEMESH_TESSELATION, (128, 128, 128))
        else:
            for i, parts in enumerate(self.connections):
                for connection in parts:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if end_idx == 10:
                        head_top_coord = self.coordinates.get(10)
                        # imp_landmark = landmarks_list.landmark[imp]
                        if head_top_coord is not None:
                            cv2.putText(img, '({},{})'.format(int(head_top_coord[0]), int(head_top_coord[1])),
                                        head_top_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), thickness=1,
                                        lineType=cv2.LINE_AA)
                    if start_idx in self.coordinates and end_idx in self.coordinates:
                        cv2.line(img, self.coordinates[start_idx], self.coordinates[end_idx], (255, 0, 0), 1)

        if self.face_attribute.counter > 3:
            rect = self.get_rect_from_landmarks(self.coordinates)
            gender, age = self.get_gender_age()
            cv2.putText(img,
                        "GENDER:{}, AGE:{}".format('MALE' if gender > 0 else 'FEMALE', int(age)),
                        (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0), thickness=1,
                        lineType=cv2.LINE_AA)

    def draw_part(self, img, coordinates, part, color):
        """ 얼굴의 지정된 파트를 그림 """
        num_landmarks = len(coordinates)
        for connection in part:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                continue
            if start_idx in coordinates and end_idx in coordinates:
                cv2.line(img, coordinates[start_idx], coordinates[end_idx], color, 1)

    def make_json(self):
        """ 얼굴 분석데이터 json 변환 """
        gender, age = self.get_gender_age()
        lips, left_eye, left_eyebrow, right_eye, right_eyebrow, face_oval = self.get_connections_from_landmarks(
            self.coordinates, self.connections)
        datas = None
        if len(face_oval) != 0:
            datas = {}
            datas['result_face'] = {}
            datas['result_face']['image_width'] = self.image_width
            datas['result_face']['image_height'] = self.image_height
            datas['result_face']['id'] = int(self.id)
            datas['result_face']['rect'] = self.get_rect_from_landmarks(self.coordinates)
            datas['result_face']['facial_attributes'] = {}
            datas['result_face']['facial_attributes']['gender'] = 'male' if gender > 0 else 'female'
            datas['result_face']['facial_attributes']['age'] = int(age)
            datas['result_face']['facial_points'] = {}
            datas['result_face']['facial_points']['lips'] = lips
            datas['result_face']['facial_points']['left_eye'] = left_eye
            datas['result_face']['facial_points']['left_eyebrow'] = left_eyebrow
            datas['result_face']['facial_points']['right_eye'] = right_eye
            datas['result_face']['facial_points']['right_eyebrow'] = right_eyebrow
            datas['result_face']['facial_points']['face_oval'] = face_oval
        # return json.dumps(datas, indent=4)
        #return json.dumps(datas)
        return datas

    def get_face_data(self, image, rect, face_attribute, draw):
        """ 성별/연령 데이터 평균 값 산출 """
        DIVIDER = 2
        face_width = rect[2] - rect[0]
        face_height = rect[3] - rect[1]
        min_x = int(rect[0] - face_width / DIVIDER)
        max_x = int(rect[2] + face_width / DIVIDER)
        min_y = int(rect[1] - face_height / DIVIDER)
        max_y = int(rect[3] + face_height / DIVIDER)

        lt_x = 0 if min_x < 0 else min_x
        lt_y = 0 if min_y < 0 else min_y
        rb_x = image.shape[1] if max_x > image.shape[1] else max_x
        rb_y = image.shape[0] if max_y > image.shape[0] else max_y

        detection_result = detect_face_image(image[lt_y:rb_y, lt_x:rb_x])

        if draw is True:
            cv2.imshow('margin_face', image[lt_y:rb_y, lt_x:rb_x])
            print(detection_result)

            for face in detection_result['result']['faces']:
                x = int(face['x'] * (rb_x - lt_x))
                w = int(face['w'] * (rb_x - lt_x))
                y = int(face['y'] * (rb_y - lt_y))
                h = int(face['h'] * (rb_y - lt_y))
                cv2.rectangle(image, (lt_x + x, lt_y + y), (lt_x + x + w, lt_y + y + h), (255, 0, 0))
                gender_female = face['facial_attributes']['gender']['female']
                gender_male = face['facial_attributes']['gender']['male']
                age = face['facial_attributes']['age']
                gender_str = 'male' if gender_male > gender_female else 'female'
                output_result = gender_str + ', ' + str(int(age))

                cv2.putText(image, output_result,
                            (rect[0], rect[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0), thickness=1,
                            lineType=cv2.LINE_AA)

        for face in detection_result['result']['faces']:
            x = int(face['x'] * (rb_x - lt_x))
            w = int(face['w'] * (rb_x - lt_x))
            y = int(face['y'] * (rb_y - lt_y))
            h = int(face['h'] * (rb_y - lt_y))
            cv2.rectangle(image, (lt_x + x, lt_y + y), (lt_x + x + w, lt_y + y + h), (255, 0, 0))
            gender_female = face['facial_attributes']['gender']['female']
            gender_male = face['facial_attributes']['gender']['male']
            age = int(face['facial_attributes']['age'])
            gender = 1 if gender_male > gender_female else -1
            face_attribute.set_gender_age(gender, age)




class ThreadFaceAttributeVariable:
    """ 얼굴 속성 분석 구조체 """
    def __init__(self):
        self.lock = threading.Lock()
        self.age = 0
        self.gender = 0
        self.counter = 0

    def set_gender_age(self, gender, age):
        self.lock.acquire()
        self.counter += 1
        try:
            if self.age == 0:
                self.age = age
            else:
                now = self.age * ((self.counter - 1) / self.counter)
                new = age * (1 / self.counter)
                # print('now:',self.age, ', new:', age)
                self.age = now + new
            self.gender += gender
        # print('gender:', self.gender)
        finally:
            # Lock을 해제해서 다른 Thread도 사용할 수 있도록 만든다.
            self.lock.release()

    # 한 Thread만 접근할 수 있도록 설정한다
    def reset(self):
        # Lock해서 다른 Thread는 기다리게 만든다.
        self.lock.acquire()
        try:
            self.age = 0
            self.gender = 0
            self.counter = 0
        finally:
            # Lock을 해제해서 다른 Thread도 사용할 수 있도록 만든다.
            self.lock.release()
