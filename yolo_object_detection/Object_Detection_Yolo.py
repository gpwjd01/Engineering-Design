import cv2
import numpy as np

# Load Yolo
# yolov3.weights : 학습된 모델, 객체를 감지하는 알고리즘의 핵싱
# yolov3.cfg : 알고리즘의 모든 설정이 있는 구성 파일
# coco.names : 알고리즘이 감지할 수 있는 개체의 이름
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
print(classes)

layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] : 아래와 같이 수정
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading Image
img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape # 이미지의 높이, 너비, 채널 받아오기

# Detecing objects 
# blob : 이미지에서 기능을 추출하고 크기를 조정하는 데 사용
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 네트워크에 넣기 위한 전처리
# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob) # 0, 1, 2 
        
net.setInput(blob) # 전처리된 blob 네트워크에 입력
outs = net.forward(output_layers) # 결과 받아오기
print(outs)

# Showing informations on the screen
class_ids = []   # 각각의 데이터를 저장할 빈 리스트 생성
confidences = [] # 각각의 데이터를 저장할 빈 리스트 생성 : 0에서 1까지의 탐지에 대한 신뢰도
boxes = []       # 각각의 데이터를 저장할 빈 리스트 생성 : 감지된 물체를 둘러싸고 있는 사각형의 좌표 
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores) # classes와 연관
        confidence = scores[class_id]
        if confidence > 0.5: # 임계값 신뢰도 : 더 크면 개체가 올바르게 감지된 것으로 간주
            # Object detected : 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            # Rectangle coordinates: 객체의 사각형 테두리 중 좌상단 좌표값 찾기
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
# print(len(boxes)) # 13
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
number_objects_detected = len(boxes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        # print(label)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+30), font, 3, color, 3)
    


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()