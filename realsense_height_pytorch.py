"""
[사람 키 측정기]
realsense의 depth거리 정보와 yolov5의 사람 검출기를 이용한 키 측정
(특정 위치에 서있어야 함)
"""


import pyrealsense2 as rs
import numpy as np
import torch
import cv2



def get_mean_height_and_distnace(height_list, distance_list, height, distnace):
    """ 키 및 거리 평균값 산출 """
    max_len = 30
    if len(height_list) > max_len:
        height_list.pop(0)
    if len(distance_list) > max_len:
        distance_list.pop(0)


    height_list.append(height)
    mean_height = sum(height_list) / len(height_list)
    distance_list.append(distnace)
    mean_distance = sum(distance_list) / len(distance_list)

    return mean_height, mean_distance

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

W = 848
H = 480

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
half_size = 0.9

print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()
height_list = []
distance_list = []


# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/

while True:
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz


    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = color_image[:, :, ::-1]
    scaled_size = (int(W), int(H))

    open_cv_image = np.array(color_image).copy()
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # center crop
    #color_image_resized = letterbox(color_image, (640, 640), stride=1)[0]

    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    results = model(color_image)
    det = results.pred[0]
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    alpha = 0.5
    blended1 = (open_cv_image * alpha) + (depth_colormap * (1 - alpha))  # 방식1
    blended1 = blended1.astype(np.uint8)  # 소수점 제거
    open_cv_image = cv2.addWeighted(open_cv_image, alpha, depth_colormap, (1 - alpha), 0)  # 방식2

    if len(det):
        # Rescale boxes from img_size to img0 size
        # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            if conf > 0.8 and cls == 0:
                bbox = [int(xyxy[0].cpu().detach()), int(xyxy[1].cpu().detach()), int(xyxy[2].cpu().detach()),
                        int(xyxy[3].cpu().detach())]

                #p1 = (int(bbox[0]), int(bbox[1]))
                #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                # x,y,z of bounding box
                obj_points = verts[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].reshape(-1, 3)
                zs = obj_points[:, 2]

                z = np.median(zs)
                # print("meter:", z)

                ys = obj_points[:, 1]
                ys = np.delete(ys, np.where(
                    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

                my = np.amin(ys, initial=1)
                #my = - 0.5
                My = np.amax(ys, initial=-1)


                height = (My - my)  # + half_size# add next to rectangle print of height using cv library
                # height = float("{:.2f}".format(height))
                # print("[INFO] object height is: ", height, "[m]")
                height_txt = str(height) + "[m]"

                mean_height, mean_distance = get_mean_height_and_distnace(height_list, distance_list, height, z)
                output_txt = 'hieght:{}({}), distance:{}({}) meters'.format(int(mean_height * 100 + 15), int(height*100), int(mean_distance * 100), int(z*100)),

                #cv2.rectangle(open_cv_image, p1, p2, (255, 0, 0), 2, 1)
                cv2.rectangle(open_cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2, 1)



                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (bbox[0], bbox[1] + 20)
                fontScale = 0.5
                fontColor = (0, 255, 0)
                lineType = 2
                cv2.putText(open_cv_image, output_txt[0],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor, thickness=1,
                            lineType=cv2.LINE_AA)

                #cv2.rectangle(open_cv_image, (n_bbox[0], n_bbox[1]), (n_bbox[2], n_bbox[3]), (0, 0, 255), thickness=1)

        cv2.imshow('img', open_cv_image)
        #cv2.imshow('depth', depth_colormap)
        key = cv2.waitKey(1)
        if key == 27:  # esc key
            break

# Stop streaming
pipeline.stop()