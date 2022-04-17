"""
[손 제스쳐 인식]
양 손의 랜드마크를 이용하여 손의 주먹과 보자기를 인식 함


"""
import math
import cv2


class HandGestureAnalyzer:
    """ 손 제스쳐 인식 클래스 """
    def __init__(self, connections, threshold_mul=50, size_mul=120):
        """ threshold 값 변경을 통해 주먹과 보자기의 구분의 임계값을 설정 (기본값 50) """
        self.coordinates = None
        self.image_width = None
        self.image_height = None
        self.grab = False
        self.is_right_hand = None
        self.connections = connections
        self.connections_max_index = max(max(max(connections)))
        self.recognition_threshold_mul = threshold_mul
        self.recognition_size_mul = size_mul

    def analyze_hand_gesture(self, landmarks_list, coordinates, image_width, image_height, is_right_hand):
        """ 손 제스쳐 인식 실행 """
        dist_sum_for_size, dist_sum_for_recognition, grab = self.get_distance_from_center(landmarks_list)
        self.coordinates = coordinates
        self.grab = grab
        self.image_width = image_width
        self.image_height = image_height
        self.is_right_hand = is_right_hand
        return dist_sum_for_size, dist_sum_for_recognition, grab

    def draw_hand_gesture(self, image, coordinates, dist_sum_for_size, dist_sum_for_recognition):
        """ 손 랜드마크 그리기 """
        self.draw_connections_from_landmarks(image, coordinates,
                                             self.connections, dist_sum_for_size,
                                             dist_sum_for_recognition)

    def get_distance_from_center(self, results_hand_landmarks):
        """ 손가락과 손바닥의 거리 계산 """
        center1 = results_hand_landmarks.landmark[0]
        center2 = results_hand_landmarks.landmark[5]
        center3 = results_hand_landmarks.landmark[17]
        center_xyz = ((center1.x + center2.x + center3.x) / 3,
                      (center1.y + center2.y + center3.y) / 3,
                      (center1.z + center2.z + center3.z) / 3)

        # [0, 5, 17] palm
        dist_sum_for_size = 0
        for idx in [0, 5, 17]:
            pt = results_hand_landmarks.landmark[idx]
            dist = math.dist(center_xyz, (pt.x, pt.y, pt.z))
            dist_sum_for_size += dist

        # [4, 8, 12, 16, 20] fingers
        dist_sum_for_recognition = 0
        for idx in range(4, 21, 4):
            pt = results_hand_landmarks.landmark[idx]
            dist = math.dist(center_xyz, (pt.x, pt.y, pt.z))
            dist_sum_for_recognition += dist

        grab_recognition = dist_sum_for_recognition * self.recognition_threshold_mul
        grab_size = dist_sum_for_size * self.recognition_size_mul
        if grab_size > grab_recognition:
            return dist_sum_for_size, dist_sum_for_recognition, True
        else:
            return dist_sum_for_size, dist_sum_for_recognition, False

    def get_connections_from_coordinates(self, coordinates, connections):
        """ 손 2D 전체 랜드마크를 손바닥, 엄지, 검지, 중지, 약지, 새끼 손가락으로 구분 """
        palm = []
        thumb = []
        index_finger = []
        middle_finger = []
        ring_finger = []
        pinky_finger = []
        center = []
        if connections:
            num_landmarks = len(coordinates)
            if num_landmarks > self.connections_max_index:
                # Draws the connections if the start and end landmarks are both visible.
                center1 = coordinates[0]
                center2 = coordinates[5]
                center3 = coordinates[17]
                center = (int((center1[0] + center2[0] + center3[0]) / 3),
                              int((center1[1] + center2[1] + center3[1]) / 3))

                for i, parts in enumerate(connections):
                    for connection in parts:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                            raise ValueError(f'Landmark index is out of range. Invalid connection '
                                             f'from landmark #{start_idx} to landmark #{end_idx}.')
                        if start_idx in coordinates and end_idx in coordinates:
                            if i == 0:  # HAND_PALM_CONNECTIONS
                                palm.append([coordinates[start_idx], coordinates[end_idx]])
                                # cv2.line(image, coordinates[start_idx], coordinates[end_idx], (255, 0, 0),1)
                            elif i == 1:  # HAND_THUMB_CONNECTIONS
                                thumb.append([coordinates[start_idx], coordinates[end_idx]])
                            elif i == 2:  # HAND_INDEX_FINGER_CONNECTIONS
                                index_finger.append([coordinates[start_idx], coordinates[end_idx]])
                            elif i == 3:  # HAND_MIDDLE_FINGER_CONNECTIONS
                                middle_finger.append([coordinates[start_idx], coordinates[end_idx]])
                            elif i == 4:  # HAND_RING_FINGER_CONNECTIONS
                                ring_finger.append([coordinates[start_idx], coordinates[end_idx]])
                            elif i == 5:  # hand_pinky_finger
                                pinky_finger.append([coordinates[start_idx], coordinates[end_idx]])


        return palm, thumb, index_finger, middle_finger, ring_finger, pinky_finger, center

    def make_json(self):
        """ 손 제스쳐 인식의 결과값을 json으로 변환 """
        palm, thumb, index_finger, middle_finger, ring_finger, pinky_finger, center = self.get_connections_from_coordinates(
            self.coordinates, self.connections)
        datas = None
        if len(palm) != 0:
            datas = {}
            datas['result_hand'] = {}
            datas['result_hand']['image_width'] = self.image_width
            datas['result_hand']['image_height'] = self.image_height
            datas['result_hand']['is_right_hand'] = self.is_right_hand
            datas['result_hand']['hand_action'] = self.grab
            datas['result_hand']['hand_points'] = {}
            datas['result_hand']['hand_points']['center'] = center
            datas['result_hand']['hand_points']['palm'] = palm
            datas['result_hand']['hand_points']['thumb'] = thumb
            datas['result_hand']['hand_points']['index_finger'] = index_finger
            datas['result_hand']['hand_points']['middle_finger'] = middle_finger
            datas['result_hand']['hand_points']['ring_finger'] = ring_finger
            datas['result_hand']['hand_points']['pinky_finger'] = pinky_finger
        # return json.dumps(datas, indent=4)
        # return json.dumps(datas)
        return datas



    def draw_connections_from_landmarks(self, image, coordinates, connections, dist_sum_for_size,
                                        dist_sum_for_recognition):
        """ 손의 각 파트들을 다른색으로 그림 """
        if connections:
            num_landmarks = len(coordinates)
            if num_landmarks > self.connections_max_index:
                # Draws the connections if the start and end landmarks are both visible.
                for i, parts in enumerate(connections):
                    for connection in parts:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        if start_idx in coordinates and end_idx in coordinates:
                            if i == 0:  # HAND_PALM_CONNECTIONS
                                color = (255, 255, 255)
                            elif i == 1:  # HAND_THUMB_CONNECTIONS
                                color = (255, 255, 0)
                            elif i == 2:  # HAND_INDEX_FINGER_CONNECTIONS
                                color = (255, 0, 255)
                            elif i == 3:  # HAND_MIDDLE_FINGER_CONNECTIONS
                                color = (0, 255, 255)
                            elif i == 4:  # HAND_RING_FINGER_CONNECTIONS
                                color = (255, 0, 0)
                            elif i == 5:  # hand_pinky_finger
                                color = (0, 255, 0)

                            cv2.line(image, coordinates[start_idx], coordinates[end_idx], color, 1)
                            cv2.putText(image, str(end_idx), coordinates[end_idx], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color,
                                        thickness=1, lineType=cv2.LINE_AA)

                center = (int((coordinates[17][0] + coordinates[5][0] + coordinates[0][0]) / 3),
                          int((coordinates[17][1] + coordinates[5][1] + coordinates[0][1]) / 3))

                dist_sum_for_recognition *= self.recognition_threshold_mul
                dist_sum_for_size *= self.recognition_size_mul
                cv2.circle(image, center, int(dist_sum_for_recognition), (0, 0, 255), 2)
                if dist_sum_for_size > dist_sum_for_recognition:
                    cv2.circle(image, center, int(dist_sum_for_size), (0, 255, 0), 2)
                else:
                    cv2.circle(image, center, int(dist_sum_for_size), (0, 0, 0), 2)
