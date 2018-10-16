from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import math
import random


def model1():
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths
	imagePaths = []
	for i in range(230,730):
		imagePaths.append('images/frame' + str(i) + '.jpg')

	for imagePath in imagePaths:
		print(imagePath)
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy()

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))

		# show the output images
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)

def readMP4():
	vidcap = cv2.VideoCapture('images/vtest.avi')
	success,image = vidcap.read()
	count = 0
	while success:
	  cv2.imwrite("images/stock1/frames%d.jpg" % count, image)     # save frame as JPEG file      
	  success,image = vidcap.read()
	  print('Read a new frame: ', success)
	  count += 1


def model2():
	color = (255,255,255)
	velocityList = []
	start = 0
	end = 0
	human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
	human_cascade2 = cv2.CascadeClassifier('haarcascade_profileface.xml')
	topLeft = []
	for i in range(0,700):
		oldTopLeft = topLeft
		topLeft = []
		start = time.time()
		img = cv2.imread('images/stock1/frames' + str(i) + '.jpg') 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		human = human_cascade.detectMultiScale(gray)
		#human2 = human_cascade2.detectMultiScale(gray)


		for (x,y,w,h) in human:
			topLeft.append(x)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
		
		#for (x,y,w,h) in human2:
			#cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)

		frames = calculateVelocity(oldTopLeft, topLeft, i)
		velocityList.append(frames)
		velocities = np.array(velocityList)
		q1 = np.percentile(velocities, 25)
		q3 = np.percentile(velocities, 75)

		if frames >= q3 + 1.5 * (q3-q1):
			color = (0,0,255)
		else:
			color = (255,255,255)


		cv2.putText(img, "velocity: " + str(frames), (10,500), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2);
		cv2.imshow('img', img)
		cv2.waitKey(60)


	cv2.destroyAllWindows()

def calculateVelocity(l1, l2, frame):
	firstSpeed = 0
	secondSpeed = 0
	for i in range(min(len(l1), len(l2))):
		if l1[i] != l2[i]:
			firstSpeed += l1[i]
			secondSpeed += l2[i]
	return abs(firstSpeed - secondSpeed) / 2


model2()