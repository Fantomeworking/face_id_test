import numpy as np
import cv2
import sys

def cap1(times):
    # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
    cap = cv2.VideoCapture(0)
    # 人脸识别的参数
    face_cascade = cv2.CascadeClassifier(
        '/home/guimu/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    index1 = 0
    index2 = 0
    Face_Train = np.zeros((((times, 3, 200, 200))))

    while index2 < times:
        # 从摄像头读取图片
        sucess, img = cap.read()

        gray = img#cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# 转为灰度图片

        # 显示摄像头，背景是灰度。
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=1,
            minSize=(5, 5)
        )

        for(x, y, w, h) in faces:
            Train = cv2.rectangle(
                gray, (x, y), (x + w, y + w), (0, 255, 0), 2)  # 矩形

            # cv2.circle(gray,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)#圆形
        # print(img[0],len(img[0]),len(img))
            # 获取识别的脸部图片
            n_x = x + w // 2 - 100
            n_y = y + w // 2 - 100
            if len(faces) > 0 and w > 200 and 150 < n_x < 490 and 140 < n_y < 340:
                
                Face_Train[index2] = Train[n_y:n_y + 200, n_x:n_x + 200].transpose(2,1,0)  # 将脸部图片200*200*3转置为3*200*200并截取
                index2 += 1
                cv2.putText(gray, str(index2), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)  # ,n_x,n_y,len(Train[n_y:n_y+200][n_x:n_x+200]))
        cv2.putText(gray, 'Esc to exit', (530, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # ,n_x,n_y,len(Train[n_y:n_y+200][n_x:n_x+200]))
        # 显示摄像头，背景是灰度。
        cv2.imshow("img", gray)
        k = cv2.waitKey(1)
        if k == 27:
            # 通过esc键退出摄像
            cv2.destroyAllWindows()
            sys.exit(0)
            break

        index1 += 1
    # 关闭摄像头
    cap.release()
    cv2.destroyAllWindows()
    return Face_Train
