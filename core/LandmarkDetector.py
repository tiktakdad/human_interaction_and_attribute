"""
[통합 랜드마크 검출기]
얼굴, 손, 몸의 뼈대의 3D 랜드마크를 검출
"""
import math
import cv2
import numpy as np
import mediapipe as mp
from core.common import get_landmarks_data, get_face_bbox
from mediapipe.python.solutions.hands_connections import HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS, \
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS, HAND_RING_FINGER_CONNECTIONS, \
    HAND_PINKY_FINGER_CONNECTIONS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, \
    FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
from core.iris_connections import LEFT_IRIS_CONNECTIONS, RIGHT_IRIS_CONNECTIONS


class LandmarkDetector:
    """ 통합 랜드마크 검출 클래스 """
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7, detection_max_size = True, iris_refining=True) -> None:
        super().__init__()
        self.face_holistic_dist_threshold = 4
        self.iris_refining = iris_refining
        self.detection_max_size = detection_max_size
        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.mp_iris = mp.solutions.iris.Iris()
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=min_detection_confidence,
                                                          min_tracking_confidence=min_tracking_confidence)
        self.hand_connections = [HAND_PALM_CONNECTIONS,
                                 HAND_THUMB_CONNECTIONS,
                                 HAND_INDEX_FINGER_CONNECTIONS,
                                 HAND_MIDDLE_FINGER_CONNECTIONS,
                                 HAND_RING_FINGER_CONNECTIONS,
                                 HAND_PINKY_FINGER_CONNECTIONS]

        self.face_connections = [FACEMESH_LIPS,
                                 FACEMESH_LEFT_EYE,
                                 FACEMESH_LEFT_EYEBROW,
                                 FACEMESH_RIGHT_EYE,
                                 FACEMESH_RIGHT_EYEBROW,
                                 FACEMESH_FACE_OVAL]

        self.iris_connections = [FACEMESH_LEFT_EYE,
                                 LEFT_IRIS_CONNECTIONS,
                                 FACEMESH_RIGHT_EYE,
                                 RIGHT_IRIS_CONNECTIONS]

    def get_rect_from_landmarks(self, landmarks_list):
        """ 랜드마크에서 바운딩박스를 추출 """
        point_list = np.array(list(landmarks_list.values()))
        x_min, y_min = point_list.min(axis=0)
        x_max, y_max = point_list.max(axis=0)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]



    def process(self, image):
        """ 랜드마크 검출기 실행 """
        if self.detection_max_size:
            results_face = self.mp_face.process(image)
            face_bbox = []
            if results_face.detections:
                for detection in results_face.detections:
                    face_bbox.append(get_face_bbox(detection, image_rows=image.shape[0], image_cols=image.shape[1]))

            if len(face_bbox) > 1:
                max_size_idx = 0
                max_size = 0
                for idx, bbox in enumerate(face_bbox): 
                    if bbox is not None:
                        if bbox[0] is not None and bbox[1]:
                            if bbox[0][0] is not None and bbox[0][1] is not None and bbox[1][0] is not None and bbox[1][1] is not None:
                                bbox_size = abs(bbox[1][0] - bbox[0][0]) * abs(bbox[1][1]-bbox[0][1])
                                if bbox_size > max_size:
                                    max_size_idx = idx
                                    max_size = bbox_size

                for idx, bbox in enumerate(face_bbox):
                    if idx != max_size_idx:
                        cv2.rectangle(image, face_bbox[idx][0], face_bbox[idx][1], (0, 0, 0), cv2.FILLED)

        results = self.mp_holistic.process(image)
        results_iris = self.mp_iris.process(image)
        face_coordinates = None
        face_landmarks = None
        left_hand_coordinates = None
        left_hand_landmarks = None
        right_hand_coordinates = None
        right_hand_landmarks = None
        iris_coordinates = None
        iris_landmarks = None
        pose_coordinates = None
        pose_landmarks = None


        if results.face_landmarks:
            idx_to_coordinates = get_landmarks_data(results.face_landmarks, image_rows=image.shape[0],
                                                    image_cols=image.shape[1])
            face_coordinates = idx_to_coordinates
            face_landmarks = results.face_landmarks

        if results.left_hand_landmarks:
            idx_to_coordinates = get_landmarks_data(results.left_hand_landmarks, image_rows=image.shape[0],
                                                    image_cols=image.shape[1])
            left_hand_coordinates = idx_to_coordinates
            left_hand_landmarks = results.left_hand_landmarks

        if results.right_hand_landmarks:
            idx_to_coordinates = get_landmarks_data(results.right_hand_landmarks, image_rows=image.shape[0],
                                                    image_cols=image.shape[1])
            right_hand_coordinates = idx_to_coordinates
            right_hand_landmarks = results.right_hand_landmarks

        if results_iris.face_landmarks_with_iris:
            idx_to_coordinates = get_landmarks_data(results_iris.face_landmarks_with_iris, image_rows=image.shape[0],
                                                    image_cols=image.shape[1], thresh_hold=False)

            dist_sum = 0
            dist_mean = 0
            if face_coordinates is not None:
                for point_idx in range(len(face_coordinates)):
                    face_iris_pt = idx_to_coordinates.get(point_idx)
                    face_pt = face_coordinates.get(point_idx)
                    if face_iris_pt is not None and face_pt is not None:
                        dist = math.dist(face_iris_pt, face_pt)
                        dist_sum += dist
                dist_mean = dist_sum/len(face_coordinates)

            if dist_mean < self.face_holistic_dist_threshold:
                if face_coordinates:
                    idx_to_coordinates.update(face_coordinates)

                iris_coordinates = idx_to_coordinates
                iris_landmarks = results_iris.face_landmarks_with_iris

                if self.iris_refining:


                    check_list = [468, 471, 33, 469, 133, 470, 159, 472, 145, 473, 474, 362, 476, 263, 475, 386, 477, 374]
                    do_it = True
                    for check in check_list:
                        if iris_coordinates.get(check) is None:
                            do_it = False
                            break
                    if do_it:
                        # check right_iris x_axis
                        right_iris = [iris_coordinates.get(468)[0], iris_coordinates.get(468)[1]]
                        if iris_coordinates.get(471)[0] < iris_coordinates.get(33)[0]:
                            right_iris[0] = right_iris[0] - abs(iris_coordinates.get(471)[0] - iris_coordinates.get(33)[0])
                        if iris_coordinates.get(469)[0] > iris_coordinates.get(133)[0]:
                            right_iris[0] = right_iris[0] + abs(iris_coordinates.get(469)[0] - iris_coordinates.get(133)[0])
                        # check right_iris y_axis
                        if iris_coordinates.get(470)[1] < iris_coordinates.get(159)[1]:
                            right_iris[1] = right_iris[1] - abs(iris_coordinates.get(470)[1] - iris_coordinates.get(159)[1])/4
                        if iris_coordinates.get(472)[1] > iris_coordinates.get(145)[1]:
                            right_iris[1] = right_iris[1] + abs(iris_coordinates.get(472)[1] - iris_coordinates.get(145)[1])/4
                        right_iris_tuple = (int(right_iris[0]), int(right_iris[1]))
                        iris_coordinates[468] = right_iris_tuple

                        # check left_iris x_axis
                        left_iris = [iris_coordinates.get(473)[0], iris_coordinates.get(473)[1]]
                        if iris_coordinates.get(474)[0] < iris_coordinates.get(362)[0]:
                            left_iris[0] = left_iris[0] - abs(iris_coordinates.get(474)[0] - iris_coordinates.get(362)[0])
                        if iris_coordinates.get(476)[0] > iris_coordinates.get(263)[0]:
                            left_iris[0] = left_iris[0] + abs(iris_coordinates.get(476)[0] - iris_coordinates.get(263)[0])
                        # check right_iris y_axis
                        if iris_coordinates.get(475)[1] < iris_coordinates.get(386)[1]:
                            left_iris[1] = left_iris[1] - abs(iris_coordinates.get(475)[1] - iris_coordinates.get(386)[1])/4
                        if iris_coordinates.get(477)[1] > iris_coordinates.get(374)[1]:
                            left_iris[1] = left_iris[1] + abs(iris_coordinates.get(477)[1] - iris_coordinates.get(374)[1])/4
                        left_iris_tuple = (int(left_iris[0]), int(left_iris[1]))
                        iris_coordinates[473] = left_iris_tuple
            else:
                #print('distance over reset.')
                self.mp_iris.reset()


        if results.pose_landmarks:
            idx_to_coordinates = get_landmarks_data(results.pose_landmarks, image_rows=image.shape[0],
                                                    image_cols=image.shape[1])
            pose_coordinates = idx_to_coordinates
            pose_landmarks = results.right_hand_landmarks

        return {'face_coordinates': face_coordinates, 'face_landmarks': face_landmarks,
                'left_hand_coordinates': left_hand_coordinates, 'left_hand_landmarks': left_hand_landmarks,
                'right_hand_coordinates': right_hand_coordinates, 'right_hand_landmarks': right_hand_landmarks,
                'iris_coordinates': iris_coordinates, 'iris_landmarks': iris_landmarks,
                'pose_coordinates': pose_coordinates, 'pose_landmarks': pose_landmarks
                }
