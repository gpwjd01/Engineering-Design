import cv2
from tracker import *

# Create tracker Object : 추적 탐지
tracker = EuclideanDistTracker() # 유클리드 거리 추적기
 
cap = cv2.VideoCapture("highway.mp4") # 파일 호출

# object detection from stable camera
# history : 길이, 기본값은 500.
# varThreshold : 픽셀과 모델 사이의 마할라노비스 거리 제곱에 대한 임계값. 
# 해당 픽셀이 배경 모델에 의해 잘 표현되는지 판단. 기본값 16. 낮을수록 false positive 발생 가능성 증가
# detecShadows : 그림자 검출 여부. 기본값은 True 
object_dector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # 배경 비율 반환 마스트 생성

while True:
    ret, frame = cap.read() # 재생되는 동영상의 한 프레임씩 읽음
    height, width, _ = frame.shape
    print(height, width)
    
    # Extract Region of interest : 관심 영역 추출
    roi = frame[340: 700, 500: 800] # height, width, 이 영역에 들어오는 것만 tracking
    
    # 1. object detection
    mask = object_dector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # clean the mask, 0~255, white
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 외관선 정보 검출
    detections = [] # 경계 상자의 각 위치 리스트
    for cnt in contours:
        # Calculate area and remove small elements 
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255,0), 2) # 모든 영상
            # 사각형 그리기 : bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h]) # 경계 상자의 각 위치 append
    
    # 2. Object Tacking
    boxes_ids = tracker.update(detections) # 각 개체에 고유 아이디 할당
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) # 15cm 위에 텍스트 출력
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    # print(boxes_ids)
    
    # print(detections) 
    cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(30) # 30ms만큼 대기
    if key == 27:
        break
    
    
cap.release() # cap 객체를 해제, 생성한 모든 윈도우 제거
cv2.destroyAllWindows() # 열린 모든 창 닫음