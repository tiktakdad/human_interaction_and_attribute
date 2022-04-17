"""
[키오스크 애플리케이션 비디오 테스트]
키오스크 애플리케이션을 단일 테스트를 위한 비디오 테스트
"""

import argparse
import cv2

from core.FaceAttributeAnalyzer import FaceAttributeAnalyzer
from core.HandGestureAnalyzer import HandGestureAnalyzer
from core.IrisGazeAnalyzer import IrisGazeAnalyzer
from core.LandmarkDetector import LandmarkDetector
from core.SocketIO import SocketClient
from core.common import draw_part
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

def test():
    """ 비디오 테스트
    -i : 서버 ip 주소
    -p : 서버 port 주소 (기본값 9999)
    -b : back-log
    -c : usb 카메라 인덱스 (기본값: 0)
    """
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

    parser.add_argument('-c',
                        '--camera-index',
                        type=int,
                        default=0,
                        help='Index of the camera')

    FLAGS = parser.parse_args()
    # Attempt connection to server

    ## usb 카메라 스트림 오픈
    cap = cv2.VideoCapture(FLAGS.camera_index)
    # For webcam input:
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    #cap = cv2.VideoCapture('D:/Videos/kiosk_sample/multi_person.mp4')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    
    ## usb 카메라 스트림 오픈 체크
    if cap.isOpened() is False:
        print('index[{}] camera is not connected to device.'.format(FLAGS.camera_index))
        return -1

    ## 카메라 이미지 입력 사이즈 고정 (기본값: 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ## 통합 랜드마크 검출기 생성
    landmark_detector = LandmarkDetector()
    ## 얼굴 분석기 생성, 성별/연령 분석 attribute_interval값 (기본값:1)초마다 수행
    face_analyzer = FaceAttributeAnalyzer(1, landmark_detector.face_connections)
    ## 손 제스쳐 인식기 생성
    hand_gesture_analyzer = HandGestureAnalyzer(landmark_detector.hand_connections)
    ## 홍채/시선 검출기 생성
    iris_gaze_analyzer = IrisGazeAnalyzer(landmark_detector.iris_connections)

    ## 클라이언트 소켓 실행(서버가 먼저 구동되 있어야 함)
    if (FLAGS.port > 0):
        client = SocketClient(FLAGS.ip_address, FLAGS.port)
        client.start()
    else:
        client = None

    face_id = 0
    missing_face = False
    is_full_screen = False
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        #image = cv2.flip(image, 1)
        draw_image = image.copy()
        ## 검출 결과값 누적
        json_results = []


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        ## 통합 랜드마크 검출기 수행
        result = landmark_detector.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 사람 스켈레톤 검출기(손 검출 정확도 향상을 위한)
        if result['pose_coordinates']:
            # 사람 몸 스켈레톤 검출 결과값 그리기
            draw_part(draw_image, result['pose_coordinates'], POSE_CONNECTIONS, (255, 255, 255))

        if result['face_coordinates']:
            # 얼굴 추적 실패하면 아이디 변경 후 누적 데이터 초기화
            if missing_face:
                face_id = 1 if face_id == 0 else 0
                missing_face = False

            ## 얼굴 분석 수행
            face_analyzer.analyze_face(draw_image, result['face_coordinates'], face_id)
            ## 얼굴 파트 그리기
            face_analyzer.draw_parts(draw_image, tessellation=True)
            ## 얼굴 분석 결과값 json 데이터 생성 및 통합에 추가
            json_results.append(face_analyzer.make_json())
        else:
            missing_face = True

        if result['iris_coordinates']:
            ## 홍채/시선 검출기 수행
            face_cube, eye_start_end, euler_angles = iris_gaze_analyzer.analyze_iris_gaze(result['iris_landmarks'],
                                                                            result['iris_coordinates'], image.shape[1],
                                                                            image.shape[0], 2, 3)

            ## 시선 시야 박스 그리기
            iris_gaze_analyzer.draw_face_cube(draw_image, face_cube, euler_angles)
            ## 홍채 그리기
            iris_gaze_analyzer.draw_iris(draw_image)
            ## 시선 시야박스와 홍채 검출이 계산된 최종 포인트 그리기
            iris_gaze_analyzer.draw_gaze(draw_image, eye_start_end, (0, 255, 0))
            ## 홍채/시선 분석 결과값 json 데이터 생성 및 통합에 추가
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
            ## 오른손 제스쳐 인식 수행
            dist_sum_for_size, dist_sum_for_recognition, _ = hand_gesture_analyzer.analyze_hand_gesture(
                result['right_hand_landmarks'],
                result['right_hand_coordinates'],
                image.shape[1],
                image.shape[0], True)
            ## 오른손 결과값 그리기
            hand_gesture_analyzer.draw_hand_gesture(draw_image, result['right_hand_coordinates'], dist_sum_for_size,
                                                    dist_sum_for_recognition)
            ## 오른손 제스쳐 분석 결과값 json 데이터 생성 및 통합에 추가
            json_results.append(hand_gesture_analyzer.make_json())

        elif result['left_hand_coordinates']:
            ## 왼손 제스쳐 인식 수행
            dist_sum_for_size, dist_sum_for_recognition, _ = hand_gesture_analyzer.analyze_hand_gesture(
                result['left_hand_landmarks'],
                result['left_hand_coordinates'],
                image.shape[1],
                image.shape[0], False)
            ## 왼손 결과값 그리기
            hand_gesture_analyzer.draw_hand_gesture(draw_image, result['left_hand_coordinates'], dist_sum_for_size,
                                                    dist_sum_for_recognition)
            ## 왼손 제스쳐 분석 결과값 json 데이터 생성 및 통합에 추가
            json_results.append(hand_gesture_analyzer.make_json())

        if client:
            ## 통합 json 결과값을 소켓통신 데이터에 추가 및 자동전송
            client.insert_data(json_results)

        #draw_image= cv2.flip(draw_image, 1)
        cv2.imshow('test', draw_image)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break
        elif key == ord('r'):
            face_id += 1
        elif key == ord('f'):
            ## 창 활성화 후, f키를 누르면 전체화면 모드로 전환. f키를 누르면 보통 창 모드로 다시 복귀
            if is_full_screen is False:
                cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_full_screen = True
            else:
                cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_full_screen = False

    cap.release()
    if client:
        client.release()


if __name__=='__main__':
    test()
    print('exit()')
