"""
[키오스크 애플리케이션 이미지폴더 테스트]
키오스크 애플리케이션을 단일 테스트를 위한 이미지 폴더 테스트
"""

import argparse
import cv2
import glob

from core.FaceAttributeAnalyzer import FaceAttributeAnalyzer
from core.HandGestureAnalyzer import HandGestureAnalyzer
from core.IrisGazeAnalyzer import IrisGazeAnalyzer
from core.LandmarkDetector import LandmarkDetector
from core.SocketIO import SocketClient
from core.common import draw_part
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

if __name__ == '__main__':
    # Get host and port
    parser = argparse.ArgumentParser(description='face analyzer client')
    parser.add_argument('-i',
                        '--ip-address',
                        type=str,
                        default='localhost',
                        help='Host ip address')
    parser.add_argument('-p',
                        '--port',
                        type=int,
                        default=-1,
                        help='Host port')
    parser.add_argument('-b',
                        '--back-log',
                        type=int,
                        default=5,
                        help='Number of backlog clients')

    FLAGS = parser.parse_args()
    #img_path_list = ['images/n_1.jpg', 'images/n_2.jpg']
    #img_path_list = ['images/11.jpg']
    #img_path_list = ['images/16.jpg', 'images/17.jpg']
    #img_path_list = ['images/21.jpg', 'images/22.jpg', 'images/23.jpg']
    #img_path_list = ['images/16.jpg', 'images/17.jpg', 'images/18.jpg', 'images/19.jpg']
    # 20211010_131518
    # 20211010_132017
    # 20211010_134816
    folder_path = 'D:/Documents/WYSWYG/20211010_134816/'
    landmark_detector = LandmarkDetector()
    face_analyzer = FaceAttributeAnalyzer(1, landmark_detector.face_connections)
    hand_gesture_analyzer = HandGestureAnalyzer(landmark_detector.hand_connections)
    iris_gaze_analyzer = IrisGazeAnalyzer(landmark_detector.iris_connections)

    for img_path in glob.glob(folder_path+'*.png'):
        # Attempt connection to server

        # For webcam input:
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        # cap = cv2.VideoCapture('D:/Videos/lck_interview.mp4')

        '''
        face_detector = FaceLandmarkDetector(output_max_face=True)
        face_analyzer = FaceAttributeAnalyzer(1, face_detector.connections)
        hand_detector = HandLandmarkDetector()
        hand_gesture_analyzer = HandGestureAnalyzer(hand_detector.connections)
        '''



        # iris_landmark_detector = IrisLandmarkDectector()
        # iris_gaze_analyzer = IrisGazeAnalyzer(iris_landmark_detector.iris_connections)

        if (FLAGS.port > 0):
            client = SocketClient(FLAGS.ip_address, FLAGS.port)
            client.start()
        else:
            client = None

        face_id = 0

        cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        #cv2.resizeWindow('show', 1024, 768)

        image = cv2.imread(img_path)
        if image is not None:
            json_results = []
            #image = cv2.flip(image, 1)
            draw_image = image.copy()

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            result = landmark_detector.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if result['pose_coordinates']:
                draw_part(draw_image, result['pose_coordinates'], POSE_CONNECTIONS, (128, 128, 128))

            if result['face_coordinates']:
                face_analyzer.analyze_face(draw_image, result['face_coordinates'], face_id)
                face_analyzer.draw_parts(draw_image, tessellation=False)
                # json_results.append(face_analyzer.make_json())
            else:
                missing_face = True

            if result['iris_coordinates']:
                face_cube, eye_start_end, euler_angles = iris_gaze_analyzer.analyze_iris_gaze(result['iris_landmarks'],
                                                                                              result[
                                                                                                  'iris_coordinates'],
                                                                                              image.shape[1],
                                                                                              image.shape[0], 2, 3)
                iris_gaze_analyzer.draw_gaze_selected_section(draw_image, eye_start_end)
                iris_gaze_analyzer.draw_face_cube(draw_image, face_cube, euler_angles)
                iris_gaze_analyzer.draw_iris(draw_image)
                iris_gaze_analyzer.draw_gaze(draw_image, eye_start_end, (0, 255, 0))
                json_results.append(iris_gaze_analyzer.make_json())
                # iris_gaze_analyzer.draw_landmarks(draw_image, result['iris_coordinates'])
                # print(right_gaze, left_gaze)
                '''
                iris_gaze_analyzer.draw_landmarks(draw_image, result['iris_coordinates'])
                iris_gaze_analyzer.draw_parts_number(draw_image, result['iris_coordinates'], (255, 255, 255))
                important_coordinates = iris_gaze_analyzer.get_important_points(
                    result['iris_landmarks'],
                    result['iris_coordinates'],
                    image.shape)
                euler_angles, start_end = iris_gaze_analyzer.get_face_angle(draw_image, important_coordinates)
                track_pt = iris_gaze_analyzer.track_gaze(start_end[1])
                track_start_end = (start_end[0], track_pt)
                iris_gaze_analyzer.draw_angle(draw_image, euler_angles, track_start_end)
                '''

            if result['right_hand_coordinates']:
                dist_sum_for_size, dist_sum_for_recognition, _ = hand_gesture_analyzer.analyze_hand_gesture(
                    result['right_hand_landmarks'],
                    result['right_hand_coordinates'],
                    image.shape[1],
                    image.shape[0], True)
                hand_gesture_analyzer.draw_hand_gesture(draw_image, result['right_hand_coordinates'], dist_sum_for_size,
                                                        dist_sum_for_recognition)

                # json_results.append(hand_gesture_analyzer.make_json())

            elif result['left_hand_coordinates']:
                dist_sum_for_size, dist_sum_for_recognition, _ = hand_gesture_analyzer.analyze_hand_gesture(
                    result['left_hand_landmarks'],
                    result['left_hand_coordinates'],
                    image.shape[1],
                    image.shape[0], False)
                hand_gesture_analyzer.draw_hand_gesture(draw_image, result['left_hand_coordinates'], dist_sum_for_size,
                                                        dist_sum_for_recognition)
                # json_results.append(hand_gesture_analyzer.make_json())

            if client:
                client.insert_data(json_results)


            cv2.imshow('show', draw_image)
            cv2.imwrite('images/output.jpg', draw_image)
            cv2.imwrite('images/output_ori.jpg', image)
            key = cv2.waitKey(1)

            if client:
                client.release()

