

import cv2
import os

# moviePath: 保存电影的路径
# dataPath: 存储截图的文件夹路径
# frameMode: 是否通过统计帧数来截图
# perSec: 多少秒一张截图（frameMode=1时无效）
# perSec: 多少帧一张截图（frameMode=0时无效）

def getScreenShotData(moviePath, dataPath="data/screenshots", frameMode=True, perSec=1.0, perFrame=23):
	if(not os.path.exists(dataPath)):
		os.makedirs(dataPath) 
	if(not os.path.isdir(dataPath)):
		return -1 
	video = cv2.VideoCapture(moviePath) #读入视频文件
	if (not video.isOpened()): #判断是否正常打开
		video.release()		
		return -1 
	currentFrame = 1
	fps = video.get(cv2.CAP_PROP_FPS)
	timeF = perFrame if frameMode else int(perSec*fps) ; #视频帧计数间隔频率
	rval = True 
	while rval:   #循环读取视频帧
		rval, frame = video.read()
		if(currentFrame%timeF == 0): #每隔timeF帧进行存储操作
			screenShotPath = os.path.join(dataPath, "screenshot%d.jpg"%currentFrame)
			cv2.imwrite(screenShotPath, frame) #存储为图像
		currentFrame = currentFrame + 1
		cv2.waitKey(1)
	video.release()