import cv2
import numpy as np
from EKFTracking import EKFTracker
from BuoyClassification import BuoyClass

COLORS = [(0, 255, 255), (0, 140, 255), (0, 255, 0)]

def NMS(pt, cache, threshold):
	'''
	Non-maximum suppression technique is commonly used to supress redundant bounding boxes at the same location
	:param pt: top left corner point of the bounding box
	:param cache: previously detected points
	:param threshold: distance threshold between two points
	:return: filter out redundant points based on the threshold value
	'''
	curr = np.array(pt)
	for ref in cache:
		if np.linalg.norm(ref - curr) < threshold:
			return True
	return False


def DetectSingleBuoy(img_gray, template):
	"""
	@brief: This function employs template matching technique to detect a single type of buoy.
	:param img_gray: Gray scale image
	:param template: Sampled colored template
	:return: box top left and bottom right coordinates
	"""
	# Store width and height of template in w and h
	w, h = template.shape[::-1]

	detection = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
	# Specify a threshold
	threshold = 0.8

	# Store the coordinates of matched area in a numpy array
	index_loc = np.where(detection >= threshold)

	for pt in zip(*index_loc[::-1]):
		box = [pt, (pt[0] + w, pt[1] + h)]
		yield box


# Convert it to grayscale
def detect_buoy(time, img_rgb, trakers):
	'''
	:param time: frame index
	:param img_rgb: colored image directly from video frame
	:param trakers: a set of EKFs
	:return: overlay bounding boxes on rgb image
	'''

	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

	# Read the template in gray scale
	template1 = cv2.imread('D:/PyCharm_project/BuoyDetection/templates/Template1.jpg', 0)
	template2 = cv2.imread('D:/PyCharm_project/BuoyDetection/templates/Template2.jpg', 0)
	template3 = cv2.imread('D:/PyCharm_project/BuoyDetection/templates/Template3.jpg', 0)
	template4 = cv2.imread('D:/PyCharm_project/BuoyDetection/templates/Template4.jpg', 0)
	template5 = cv2.imread('D:/PyCharm_project/BuoyDetection/templates/Template5.jpg', 0)

	cache = []
	for i, template in enumerate([template1, template2, template3, template4, template5]):
		# perform template matching to detect buoy based on shape information

		for box in DetectSingleBuoy(img_gray.copy(), template):
			pt = box[0]
			rect = np.array(box, dtype=int)
			size = rect[1] - rect[0]

			if len(cache) != 0 and  NMS(pt, cache, size[0]):
				continue
			cache.append(np.array(pt))

			# generate image patch by cropping rgb image

			pos = np.array(rect[0], dtype=int)
			patch = img_rgb[rect[0][1] : rect[0][1] + size[1], rect[0][0] : rect[0][0] + size[0]]

			# classify image patch based on its color dominance
			classID = -1
			for key in BuoyClass(patch):
				classID = key

			if time == 0:
				# initialize EKF with initial box location at the beginning
				trakers[classID].set(pos)
			# update EKF
			trakers[classID].update(pos)

			# draw colored bounding box
			trackedBox = trakers[classID].get()
			trackedBox = np.squeeze(trackedBox)
			cv2.rectangle(img_rgb, (trackedBox[0], trackedBox[1]), (trackedBox[0] + size[0], trackedBox[1] + size[1]), COLORS[classID], 2)

	# Show the final image with the matched area.
	cv2.imshow('Detected', img_rgb)
	cv2.waitKey(10)
	return  img_rgb

if __name__ == "__main__":

	cap = cv2.VideoCapture("data/detectbuoy.avi")
	trakers = [EKFTracker(), EKFTracker(), EKFTracker()]
	time = 0

	out = cv2.VideoWriter('results/tracking_t5.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Error opening video stream or file")

	# Read until video is completed
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:

			# Display the resulting frame
			# cv2.imshow('Frame', frame)
			detect_buoy(time, frame, trakers)
			out.write(frame)
			time += 1
			# Press Q on keyboard to  exit
			if cv2.waitKey(100) & 0xFF == ord('q'):
				break

		# Break the loop
		else:
			break

	# When everything done, release the video capture object
	cap.release()
	# save result as a video format
	out.release()

	# Closes all the frames
	cv2.destroyAllWindows()