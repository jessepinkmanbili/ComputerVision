import cv2 as cv
import os

os.chdir("E:\Syspetro\_21_CodeLearning\Python\ComputerVision\FaceRecognization")

# face_detector
def face_detector_method(path):
    # 转为灰度图
    # img = cv.imread(path)
    # img_gray = img
    img_gray = cv.cvtColor(path, cv.COLOR_BGRA2GRAY)
    # 定义分类器，使用opencv自带的分类器
    face_detector = cv.CascadeClassifier('D:\SortWare\Anaconda\InstallSite\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    # 使用分类器、窗口比例系数（scaleFactor）、至少被检测次数（minNeighbors）
    face = face_detector.detectMultiScale(img_gray, 1.1, 5, 0, (10, 10), (500, 500))
    # 画矩阵
    for x,y,w,h in face:
        cv.rectangle(img_gray, (x, y), (x+w, y+h), (255, 0, 0), thickness = 2)
    cv.imshow("result", img_gray)
    
    return img_gray
    # cv.waitKey(0)
    # cv.destroyAllWindows()

face_detector_method("cat.jpg")
# 多人脸识别
face_detector_method("faces.jpg")

# vadio_detector
# 0表示电脑自带摄像头，1为外接摄像头，也可用视频路径
cap = cv.VideoCapture(0)

num: int = 1
while cap.isOpened():
    # flag为True则读取到帧数， falme：一帧图像
    flag, frame = cap.read()
    # 视频结束，跳出循环
    if not flag:
        break
    # 视频帧人脸检测
    img_gray = face_detector_method(frame)
    # 捕捉视频帧一毫秒，设置为0只捕捉第一帧（等待无限长直到键盘输入），播放不了视频
    if cv.waitKey(10) & 0xFF == ord('c'):
        break
    
    # 在1000ms内根据键盘输入返回一个值
    if cv.waitKey(10) & 0xFF == ord('s'):
        cv.imwrite("People"+str(num)+".face"+".jpg", img_gray)
        print("Success save" + str(num) + "picture")
        print("===============================")
        num += 1
        
cv.destroyAllWindows()    
# 释放图像
cap.release()
