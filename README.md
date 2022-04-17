# WYSIWYG AI ENGINE

WYSIWYG AI engine on CPU and test client for the kiosk application

## Installation

Use the [Anaconda](https://www.anaconda.com/products/individual) and package manager [pip](https://pip.pypa.io/en/stable/) to install WYSIWYG AI engine.

```bash
conda create -n kiosk_app python=3.9
conda activate kiosk_app
pip install opencv-python
pip install requests
pip install pillow
pip install -r requirements.txt
conda install protobuf
conda install bazel
```

## Usage

```bash
python video_test.py

(or with local-server)
python video_test.py -i 127.0.0.1 -p 9999 -c 0

(or make batch-file)
set root=C:\Users\conia\anaconda3
call %root%\Scripts\activate.bat %root%
call conda env list
call conda activate kiosk_test
call cd C:\Users\conia\Documents\test\kiosk_application
call python video_test.py -i 127.0.0.1 -p 9999 -c 0
pause

```

## Output JSON data format
### - FACE
![face](../image/json_face.png)
```
datas['result_face']['image_width']  : int,  이미지 가로 
datas['result_face']['image_height'] : int,  이미지 세로 
datas['result_face']['id'] =         : int,  객체 아이디
datas['result_face']['rect']         : int array[4] - lt(x,y), rb(x,y) - 얼굴 영역
datas['result_face']['facial_attributes']['gender'] : str, "male" or "female" - 남성/여성
datas['result_face']['facial_attributes']['age'] : int, 나이
datas['result_face']['facial_points']['lips']          : int array[40] - [(x,y),(x,y)] - 입술 연결선
datas['result_face']['facial_points']['left_eye']      : int array[16] - [(x,y),(x,y)] - 왼쪽 눈 연결선
datas['result_face']['facial_points']['left_eyebrow']  : int array[8] - [(x,y),(x,y)]  - 왼쪽 눈썹 연결선
datas['result_face']['facial_points']['right_eye']     : int array[16] - [(x,y),(x,y)] - 오른쪽 눈 연결선
datas['result_face']['facial_points']['right_eyebrow'] : int array[8] - [(x,y),(x,y)]  - 오른쪽 눈썹 연결선
datas['result_face']['facial_points']['face_oval']     : int array[36] - [(x,y),(x,y)] - 얼굴 윤곽 연결선
```

### - HAND
![hand](../image/json_hand.png)
```
datas['result_hand']['image_width']   : int,  이미지 가로
datas['result_hand']['image_height']  : int,  이미지 세로
datas['result_hand']['is_right_hand'] : bool,  검출된 손 - true=오른손, false=왼손
datas['result_hand']['hand_action']   : bool,  주먹/보자기 - true=주먹, false=보자기
datas['result_hand']['hand_points']['center']        : int array[2],  손 중심 위치(x,y)
datas['result_hand']['hand_points']['palm']          : int array[6],  손바닥 연결 선 ([x,y], [x,y])
datas['result_hand']['hand_points']['thumb']         : int array[3],  엄지 연결 선 ([x,y], [x,y])
datas['result_hand']['hand_points']['index_finger']  : int array[4],  검지 연결 선 ([x,y], [x,y])
datas['result_hand']['hand_points']['middle_finger'] : int array[3],  중지 연결 선 ([x,y], [x,y])
datas['result_hand']['hand_points']['ring_finger']   : int array[3],  약지 연결 선 ([x,y], [x,y])
datas['result_hand']['hand_points']['pinky_finger']  : int array[3],  새끼 연결 선 ([x,y], [x,y])
```

### - IRIS/GAZE
![iris](../image/json_iris.png)
```
datas['result_iris']['image_width']       : int,  이미지 가로
datas['result_iris']['image_height']      : int,  이미지 세로
datas['result_iris']['gaze_max_width']    : int,  맵핑 대상의 화면 가로
datas['result_iris']['gaze_max_height']   : int,  맵핑 대상의 화면 세로
datas['result_iris']['gaze_direction']    : int array[2], 응시하는 셀 (행,렬)
datas['result_iris']['gaze_point']        : int array[2], 응시하는 좌표 (x,y) 
datas['result_iris']['head_euler_angles'] : float array[3], 머리의 [pitch, yaw, roll]
datas['result_iris']['iris_points']['left_iris']  : int array[5], 왼쪽   홍채 lrtbc (x,y)좌표
datas['result_iris']['iris_points']['right_iris'] : int array[5], 오른쪽 홍채 lrtbc (x,y)좌표
```

## License
[APACHE LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
(commercial free)