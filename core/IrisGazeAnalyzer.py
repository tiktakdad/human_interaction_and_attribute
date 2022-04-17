"""
[시선 인식]
홍채 검출 및 시선 인식
"""
import cv2
import numpy as np


class IrisGazeAnalyzer:
    """ 홍채/시선 인식 클래스. depth 변수를 통해 키오스크와의 거리를 설정해줘야 함"""
    def __init__(self, connections) -> None:
        super().__init__()
        self.connections = connections
        #self.connections_max_index = max(max(max(connections)))
        self.coordinates = None
        self.landmarks_list = None
        self.image_width = None
        self.image_height = None
        self.gaze_max_width = None
        self.gaze_max_height = None
        self.is_track_eye_init = False
        self.d_center = None
        self.t_center = None
        self.col_num = None
        self.row_num = None
        self.gaze_direction = None
        self.is_track_cube_init = False
        self.d_center_cube = None
        self.t_center_cube = None
        self.track_speed = 4
        self.depth = 800
        self.view_width = 800
        self.view_height = 450
        self.translation_threshold= 10000
        self.eye_width_mul = 0.5
        self.eye_height_mul = 1.0
        self.eye_y_calibration = 0.0
        self.head_euler_angles = []
        self.gaze_point = None

    def get_iris_points(self):
        """ 왼쪽눈과 오른쪽눈의 홍채 랜드마크 좌표를 얻음"""
        # eye left, right, top, bottom, center
        # inner
        right_eye_lrtbc = np.array([self.coordinates.get(33), self.coordinates.get(133),
                                    self.coordinates.get(159), self.coordinates.get(145), self.coordinates.get(468)])
        left_eye_lrtbc = np.array([self.coordinates.get(362), self.coordinates.get(263),
                                   self.coordinates.get(386), self.coordinates.get(374), self.coordinates.get(473)])
        '''
        # outer
        right_eye_lrtbc = np.array([self.coordinates.get(226), self.coordinates.get(243),
                                    self.coordinates.get(27), self.coordinates.get(23), self.coordinates.get(468)])
        left_eye_lrtbc = np.array([self.coordinates.get(463), self.coordinates.get(446),
                                   self.coordinates.get(257), self.coordinates.get(253), self.coordinates.get(473)])
                                   '''

        for right, left in zip(right_eye_lrtbc, left_eye_lrtbc):
            if right is None or left is None:
                return None
            '''
            else:
                cv2.circle(image, right, 3, (255, 0, 0))
                cv2.circle(image, left, 3, (0, 0, 255))
            '''
        return right_eye_lrtbc, left_eye_lrtbc

    def refine_iris_coordinate(self, left_eye_lrtbc, right_eye_lrtbc):
        """ 홍채 인식의 정확도 개선을 위해 세부 위치 조정 """
        '''
               #origin
               right_width = abs(right_eye_lrtbc[0][0] - right_eye_lrtbc[1][0]) * self.eye_width_mul
               right_x_center = abs(right_eye_lrtbc[0][0] + right_eye_lrtbc[1][0]) / 2
               right_x_gaze = (right_eye_lrtbc[4][0] - right_x_center) / (right_width / 2)
               left_width = abs(left_eye_lrtbc[0][0] - left_eye_lrtbc[1][0]) * self.eye_width_mul
               left_x_center = abs(left_eye_lrtbc[0][0] + left_eye_lrtbc[1][0]) / 2
               left_x_gaze = (left_eye_lrtbc[4][0] - left_x_center) / (left_width / 2)

               right_height = abs(right_eye_lrtbc[2][1] - right_eye_lrtbc[3][1]) * self.eye_height_mul
               right_y_center = abs(right_eye_lrtbc[2][1] + right_eye_lrtbc[3][1]) / 2
               right_y_gaze = (right_eye_lrtbc[4][1] - right_y_center) / (right_height / 2) + self.eye_y_calibration
               left_height = abs(left_eye_lrtbc[2][1] - left_eye_lrtbc[3][1]) * self.eye_height_mul
               left_y_center = abs(left_eye_lrtbc[2][1] + left_eye_lrtbc[3][1]) / 2
               left_y_gaze = (left_eye_lrtbc[4][1] - left_y_center) / (left_height / 2) + self.eye_y_calibration

               cv2.circle(image, (int(right_x_center), int(right_y_center)), 3, (0, 255, 0))
               cv2.circle(image, (int(left_x_center), int(left_y_center)), 3, (0, 255, 0))
               '''

        # refined
        right_width = abs(right_eye_lrtbc[0][0] - right_eye_lrtbc[1][0]) * self.eye_width_mul
        right_x_center = abs(right_eye_lrtbc[2][0] + right_eye_lrtbc[3][0]) / 2
        right_x_gaze = (right_eye_lrtbc[4][0] - right_x_center) / (right_width / 2)
        left_width = abs(left_eye_lrtbc[0][0] - left_eye_lrtbc[1][0]) * self.eye_width_mul
        left_x_center = abs(left_eye_lrtbc[2][0] + left_eye_lrtbc[3][0]) / 2
        left_x_gaze = (left_eye_lrtbc[4][0] - left_x_center) / (left_width / 2)

        right_height = abs(right_eye_lrtbc[2][1] - right_eye_lrtbc[3][1]) * self.eye_height_mul
        right_y_center = abs(right_eye_lrtbc[2][1] + right_eye_lrtbc[3][1]) / 2
        right_y_gaze = (right_eye_lrtbc[4][1] - right_y_center) / (right_height / 2) + self.eye_y_calibration
        left_height = abs(left_eye_lrtbc[2][1] - left_eye_lrtbc[3][1]) * self.eye_height_mul
        left_y_center = abs(left_eye_lrtbc[2][1] + left_eye_lrtbc[3][1]) / 2
        left_y_gaze = (left_eye_lrtbc[4][1] - left_y_center) / (left_height / 2) + self.eye_y_calibration

        right_x_gaze = -1.0 if right_x_gaze < -1.0 else right_x_gaze
        right_x_gaze = 1.0 if right_x_gaze > 1.0 else right_x_gaze
        right_y_gaze = -1.0 if right_y_gaze < -1.0 else right_y_gaze
        right_y_gaze = 1.0 if right_y_gaze > 1.0 else right_y_gaze
        left_x_gaze = -1.0 if left_x_gaze < -1.0 else left_x_gaze
        left_x_gaze = 1.0 if left_x_gaze > 1.0 else left_x_gaze
        left_y_gaze = -1.0 if left_y_gaze < -1.0 else left_y_gaze
        left_y_gaze = 1.0 if left_y_gaze > 1.0 else left_y_gaze

        right_gaze = (right_x_gaze, right_y_gaze)
        left_gaze = (left_x_gaze, left_y_gaze)

        # mean
        output_gaze = ((right_gaze[0] + left_gaze[0]) / 2, (right_gaze[1] + left_gaze[1]) / 2)
        return output_gaze

    def analyze_iris_gaze(self, landmarks_list, coordinates, image_width, image_height, col_num, row_num):
        """ 홍채 검출 및 시선 인식 수행 """
        self.coordinates = coordinates
        self.landmarks_list = landmarks_list
        self.image_width = image_width
        self.image_height = image_height
        self.gaze_max_width = 1020
        self.gaze_max_height = 1980
        self.col_num = col_num
        self.row_num = row_num
        track_cube_start_end = None
        track_gaze_start_end = None
        euler_angles = []

        if self.get_iris_points() is None:
            return track_cube_start_end, track_gaze_start_end, euler_angles
        right_eye_lrtbc, left_eye_lrtbc = self.get_iris_points()

        output_gaze = self.refine_iris_coordinate(left_eye_lrtbc, right_eye_lrtbc)

        #cv2.circle(image, (int(right_x_center), int(right_y_center)), 3, (255, 0, 255))
        #cv2.circle(image, (int(left_x_center), int(left_y_center)), 3, (255, 0, 255))






        #max
        #max_gaze_x = right_gaze[0] if abs(right_gaze[0]) > abs(left_gaze[0]) else left_gaze[0]
        #max_gaze_y = right_gaze[1] if abs(right_gaze[1]) > abs(left_gaze[1]) else left_gaze[1]
        #output_gaze = (output_gaze[0], max_gaze_y)
        important_coordinates = self.get_important_points(coordinates)
        euler_angles, face_start_end, eye_start_end = self.get_face_angle(important_coordinates, output_gaze)
        if euler_angles is not None:
            self.head_euler_angles = euler_angles
        if eye_start_end is not None:
            track_pt = self.track_gaze(eye_start_end)
            track_start_end = (eye_start_end[0], track_pt[1])
            track_gaze_start_end = track_start_end
            self.gaze_point = track_gaze_start_end
        if face_start_end is not None:
            refine_face_start_end = self.track_cube(face_start_end)
            refine_face_start_end[:4, :, :] = face_start_end[:4, :, :]
            track_cube_start_end = refine_face_start_end
        if track_gaze_start_end is not None:
            cell_width = image_width / col_num
            cell_height = image_height / row_num
            gaze_x = track_gaze_start_end[1][0]
            gaze_y = track_gaze_start_end[1][1]
            contain_idx = None
            for i in range(0, 3):
                for j in range(0, 3):
                    if gaze_x >= int(cell_width * i) and gaze_x <= int(cell_width * (i + 1)) and gaze_y >= int(
                            cell_height * j) and gaze_y <= int(cell_height * (j + 1)):
                        contain_idx = [j, i]
                        self.gaze_direction = contain_idx
        if euler_angles is not None:
            self.euler_angles = euler_angles
        return track_cube_start_end, track_gaze_start_end, euler_angles

    def track_cube(self, cube):
        """ 사람의 시선이 어디를 보고 있는지 얼굴의 각도 변화(pitch, yaw, roll)를 통해 시야 박스를 추적 """
        if cube is not None:
            n_center = np.array(cube)
            if self.is_track_cube_init is False:
                self.t_center_cube = n_center
                self.d_center_cube = n_center
                self.is_track_cube_init = True
            self.d_center_cube = n_center - self.t_center_cube
            self.t_center_cube = self.t_center_cube + self.d_center_cube / self.track_speed
            self.t_center_cube = self.t_center_cube.round().astype(np.int)
        return self.t_center_cube

    def track_gaze(self, point):
        """ 사람의 시선이 어디를 보고 있는지 얼굴의 각도 변화와 홍채의 위치 변화를 추적"""
        if point is not None:
            n_center = np.array(point)
            n_center[1][0] = 0 if n_center[1][0] < 0 else n_center[1][0]
            n_center[1][0] = self.gaze_max_width if n_center[1][0] > self.gaze_max_width else n_center[1][0]
            n_center[1][1] = 0 if n_center[1][1] < 0 else n_center[1][1]
            n_center[1][1] = self.gaze_max_height if n_center[1][1] > self.gaze_max_height else n_center[1][1]
            if self.is_track_eye_init is False:
                self.t_center = n_center
                self.d_center = n_center
                self.is_track_eye_init = True
            self.d_center = n_center - self.t_center
            self.t_center = self.t_center + self.d_center / (self.track_speed*2)
            self.t_center = self.t_center.round().astype(np.int)
        #new_x = self.t_center[0] if self.t_center[0] < self.image_width - self.t_center[0] else self.image_width - self.t_center[0]
        #new_y = self.t_center[1] if self.t_center[1] < self.image_height - self.t_center[1] else self.image_height - self.t_center[1]
        return self.t_center

    def get_important_points(self, coordinates):
        """ 얼굴의 중요 포인트 그리기 """
        # Nose tip : 4
        # Left Mouth corner : 61
        # Left eye left corner :  130
        # Chin : 152
        # Right mouth corner : 291
        # Right eye right corner : 359
        important_points = [4, 61, 130, 152, 291, 359]
        important_landmarks_list = []
        important_coordinates = []
        for imp in important_points:
            imp_coord = coordinates.get(imp)
            # imp_landmark = landmarks_list.landmark[imp]
            if imp_coord is not None:
                important_coordinates.append(coordinates[imp])

        return important_coordinates

    def draw_face_cube(self, image, cube, euler_angles):
        """ 시선 시야 박스 그리기 """
        if cube is not None:
            imgpts = np.int32(cube).reshape(-1, 2)
            # draw ground floor in green
            img = cv2.drawContours(image, [imgpts[:4]], -1, (0, 255, 0), 1)
            # draw pillars in blue color
            for i, j in zip(range(4), range(4, 8)):
              img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 255), 1)
            # draw top layer in red color
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
            cv2.putText(image, '{0:.2f}, {1:.2f}, {2:.2f}'.format(float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])),
                        imgpts[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), thickness=1, lineType=cv2.LINE_AA)

    def draw_gaze_selected_section(self, image, start_end):
        """ 시선이 어디를 보고 있는지 n분할하여 위치 표시 """
        if start_end is not None:
            cell_width = image.shape[1] / self.col_num
            cell_height = image.shape[0] / self.row_num
            gaze_x = start_end[1][0]
            gaze_y = start_end[1][1]
            contain_idx = None
            for i in range(0, 3):
                for j in range(0, 3):
                    if gaze_x >= int(cell_width * i) and gaze_x <= int(cell_width * (i + 1)) and gaze_y >= int(
                            cell_height * j) and gaze_y <= int(cell_height * (j + 1)):
                        contain_idx = [j,i]

                    cv2.rectangle(image, (int(cell_width*i), int(cell_height*j)),
                                        (int(cell_width*(i+1)), int(cell_height*(j+1))),
                                        (128,128,128))

            if contain_idx is not None:
                cv2.rectangle(image, (int(cell_width * contain_idx[1]), int(cell_height * contain_idx[0])),
                              (int(cell_width * (contain_idx[1] + 1)), int(cell_height * (contain_idx[0] + 1))),
                              (255, 255, 255), thickness=2)



    def draw_iris(self, image):
        """ 홍채 검출 결과 그리기 """
        if self.get_iris_points() is not None:
            right_eye_lrtbc, left_eye_lrtbc = self.get_iris_points()
            self.refine_iris_coordinate(left_eye_lrtbc, right_eye_lrtbc)
            for right, left in zip(right_eye_lrtbc, left_eye_lrtbc):
                cv2.circle(image, right, 3, (255, 0, 0))
                cv2.circle(image, left, 3, (0, 0, 255))

            cv2.circle(image, right_eye_lrtbc[4], 5, (255, 255, 255))
            cv2.circle(image, left_eye_lrtbc[4], 5, (255, 255, 255))

        # cv2.circle(image, (int(right_x_center), int(right_y_center)), 3, (255, 0, 255))
        # cv2.circle(image, (int(left_x_center), int(left_y_center)), 3, (255, 0, 255))



    def draw_gaze(self, image, start_end, bgr):
        """ 시선과 홍채검출이 계산된 포인트 그리기 """
        if start_end is not None:
            cv2.line(image, start_end[0], start_end[1], bgr, 2)
            cv2.circle(image, start_end[1], 4, (0, 0, 255), 3)
            cv2.putText(image, '({},{})'.format(int(start_end[1][0]), int(start_end[1][1])),
                        start_end[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)


    def get_face_angle(self, important_coordinates, gaze=None):
        """ 오일러 앵글을 통한 얼굴의 pitch, yaw, roll 값을 계산 """
        euler_angles = []
        cube_point2D = None
        eye_point2D = None
        # cv2.line(image, p1, p2, (255, 0, 0), 2)


        if len(important_coordinates) > 5:
            size = self.image_height, self.image_width
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            image_points = np.array(important_coordinates, dtype="double")
            # image_points = np.array(landmarks, dtype="double")
            # image_points = image_points[:,:2]
            # model_points = np.array(landmarks, dtype="double")

            # print("Camera Matrix :\n {0}".format(camera_matrix))

            # Nose tip : 4
            # Left Mouth corner : 61
            # Left eye left corner :  130
            # Chin : 152
            # Right mouth corner : 291
            # Right eye right corner : 359

            model_points = np.array(
                [
                    (0.0, 0.0, 0.0),  # nose tip
                    (-150.0, -150.0, -125.0),  # left mouth corner
                    (-165.0, 170.0, -135.0),  # left eye left corner
                    (0.0, -330.0, -65.0),  # chin
                    (150.0, -150.0, -125.0),  # right mouth corner
                    (165.0, 170.0, -135.0)  # right eye right corner
                ]
            )
            '''
            model_points = np.array(
                [
                    (0.0, 0.0, 0.0),  # nose tip
                    (-150.0, -150.0, -125.0),  # left mouth corner
                    (-225.0, 170.0, -135.0),  # left eye left corner
                    (0.0, -330.0, -65.0),  # chin
                    (150.0, -150.0, -125.0),  # right mouth corner
                    (225.0, 170.0, -135.0)  # right eye right corner
                ]
            )
            '''

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            sight_size = int(self.depth / 800) if self.depth / 800 > 0 else 1
            cube = np.float32([[-self.view_width / 2, 0, 0], [-self.view_width / 2, self.view_height / 2, 0],
                               [self.view_width / 2, self.view_height / 2, 0], [self.view_width / 2, 0, 0],
                               [-self.view_width*sight_size, -self.view_height*sight_size, self.depth],
                               [-self.view_width*sight_size, self.view_height*sight_size, self.depth],
                               [self.view_width*sight_size, self.view_height*sight_size, self.depth],
                               [self.view_width*sight_size, -self.view_height*sight_size, self.depth]])

            #cube = np.array([(0.0, 0.0, 1000.0)])


            (cube_point2D, jacobian) = cv2.projectPoints(cube, rotation_vector,translation_vector, camera_matrix, dist_coeffs)


            #(nose_end_point2D, jacobian) = cv2.projectPoints(cube, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            '''
            cube_p1 = (int(image_points[0][0]), int(image_points[0][1]))
            cube_p2 = (int(cube_point2D[0][0][0]), int(cube_point2D[0][0][1]))
            cube_point2D = (cube_p1, cube_p2)
            '''
            '''
            for p in image_points:
                cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                '''



            if gaze is not None:
                # ignore y-axis gaze moving
                #eye_pt = np.array([(gaze[0]*self.view_width, -gaze[1]*self.view_height, int(self.depth))])
                eye_pt = np.array([(gaze[0] * self.view_width *sight_size, 0, int(self.depth))])
                (eye_point2D, _) = cv2.projectPoints(eye_pt, rotation_vector, translation_vector,
                                                                 camera_matrix, dist_coeffs)
                if eye_point2D is not None:
                    # 2,5
                    eye_center_x = (image_points[2][0] + image_points[5][0])/2
                    eye_center_y = (image_points[2][1] + image_points[5][1])/2
                    #eye_p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    eye_p1 = (int(eye_center_x), int(eye_center_y))
                    if np.isnan(eye_point2D[0][0][0]):
                        eye_point2D[0][0][0] = 0
                    if np.isnan(eye_point2D[0][0][1]):
                        eye_point2D[0][0][1] = 0
                    eye_p2 = (int(eye_point2D[0][0][0]), int(eye_point2D[0][0][1]))
                    eye_point2D = (eye_p1, eye_p2)

            '''
            for idx in range(len(important_coordinates)):
                cv2.circle(image, important_coordinates[idx], 1, (0, 255, 0))
                cv2.putText(image, str(idx), important_coordinates[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                            '''

            rmat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rmat, translation_vector))

            '''
            imgpts = np.int32(nose_end_point2D).reshape(-1, 2)
            # draw ground floor in green
            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
            # draw pillars in blue color
            for i, j in zip(range(4), range(4, 8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 255), 3)
            # draw top layer in red color
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
            '''

            # euler_angles contain (pitch, yaw, roll)
            # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            if abs(translation_vector[2][0]) < 0:
                print(abs(translation_vector[2][0]))
                euler_angles = []
                cube_point2D = None
                eye_point2D = None
            '''
            if abs(translation_vector[2][0]) > self.translation_threshold:
                print(abs(translation_vector[2][0]))
                euler_angles = []
                cube_point2D = None
                eye_point2D = None
                '''


        return list(euler_angles), cube_point2D, eye_point2D

    def draw_parts_number(self, img, coordinates, color):
        """ 얼굴 및 홍채 파트를 그림 """
        # Nose tip : 4
        # Left Mouth corner : 61
        # Left eye left corner :  130
        # Chin : 152
        # Right mouth corner : 291
        # Right eye right corner : 359
        important_points = []
        for idx in coordinates:
            cv2.circle(img, coordinates[idx], 1, color)
            cv2.putText(img, str(idx), coordinates[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, thickness=1, lineType=cv2.LINE_AA)

        for imp in important_points:
            imp_coord = coordinates.get(imp)
            # imp_landmark = landmarks_list.landmark[imp]
            if imp_coord is not None:
                cv2.circle(img, coordinates[imp], 1, (0, 0, 255))
                cv2.putText(img, str(imp), coordinates[imp], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255),
                            thickness=1, lineType=cv2.LINE_AA)

    def draw_landmarks(self, image, coordinates, draw_number):
        """ 전체 랜드마크 데이터를 그림 """
        if self.connections:
            num_landmarks = len(coordinates)
            #if num_landmarks > self.connections_max_index:
                # Draws the connections if the start and end landmarks are both visible.
            for i, parts in enumerate(self.connections):
                for connection in parts:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx in coordinates and end_idx in coordinates:
                        if i < 2:  # LEFT_IRIS_CONNECTIONS
                            color = (203, 192, 255)
                        else:
                            color = (255, 255, 0)
                        cv2.line(image, coordinates[start_idx], coordinates[end_idx], color, 1)

                        if draw_number:
                            cv2.putText(image, str(start_idx), coordinates[start_idx], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color,
                                        thickness=1, lineType=cv2.LINE_AA)
                            cv2.putText(image, str(end_idx), coordinates[end_idx], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color,
                                        thickness=1, lineType=cv2.LINE_AA)



    def make_json(self):
        """ 시선 분석의 결과를 json 형태로 변환 """
        datas = None
        if self.get_iris_points() is not None and len(self.head_euler_angles) > 2 and self.gaze_point is not None:
            right_eye_lrtbc, left_eye_lrtbc = self.get_iris_points()
            self.refine_iris_coordinate(left_eye_lrtbc, right_eye_lrtbc)
            datas = {}
            datas['result_iris'] = {}
            datas['result_iris']['image_width'] = self.image_width
            datas['result_iris']['image_height'] = self.image_height
            datas['result_iris']['gaze_max_width'] = self.gaze_max_width
            datas['result_iris']['gaze_max_height'] = self.gaze_max_height
            datas['result_iris']['gaze_direction'] = self.gaze_direction
            datas['result_iris']['gaze_point'] = self.gaze_point[1].tolist()
            datas['result_iris']['head_euler_angles'] = [self.head_euler_angles[0][0], self.head_euler_angles[1][0], self.head_euler_angles[2][0]]
            datas['result_iris']['iris_points'] = {}
            datas['result_iris']['iris_points']['left_iris'] = left_eye_lrtbc.tolist()
            datas['result_iris']['iris_points']['right_iris'] = right_eye_lrtbc.tolist()

        # return json.dumps(datas, indent=4)
        # return json.dumps(datas)
        return datas