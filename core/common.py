"""
[공용 모듈]
검출엔진에서 공통적으로 사용되는 모듈 집합
"""

import cv2

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates, _PRESENCE_THRESHOLD, \
    _VISIBILITY_THRESHOLD


def get_face_bbox(detection, image_rows, image_cols):
    """ 얼굴 검출 위치 데이터에서 바운딩 박스 추출 """
    if not detection.location_data:
        return

    location = detection.location_data
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
        image_rows)
    return [rect_start_point, rect_end_point]


def get_landmarks_data(landmark_list: landmark_pb2.NormalizedLandmarkList, image_rows, image_cols, thresh_hold=True):
    """ 3D 랜드마크 데이터를 2D 랜드마크 좌표로 변환  """
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if thresh_hold is True:
            if ((landmark.HasField('visibility') and
                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < _PRESENCE_THRESHOLD)):
                continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)

        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    return idx_to_coordinates

def draw_part(img, coordinates, part, color):
    """ 2D 랜드마크 좌표들의 연결을 선으로 그림  """
    num_landmarks = len(coordinates)
    for connection in part:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            continue
        if start_idx in coordinates and end_idx in coordinates:
            cv2.line(img, coordinates[start_idx], coordinates[end_idx], color, 1)