from picamera import PiCamera
import io
import numpy as np
import time
import cv2
from calibrator import ChessboardInfo, Patterns
import yaml
import math

'''
Takes shape of matrix and the data, forms a properly shaped array
'''
def buildCameraMatrix(cols, rows, data):
	my_list = []
	#Check that the data is the expected length 
	if (cols*rows != len(data)):
		return None

	for i in range(rows):
		l = []
		for j in range(cols):
			l.append(data[len(my_list)*cols + j])
		my_list.append(l)
	return np.array(my_list)

def calc_pattern_points(pattern_size, size):
	pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
	pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
	pattern_points*= size #square size
	return pattern_points

#Read camera calibration file
s = open("calibration_parameters")
settings = yaml.load(s)

raw_camera_matrix = np.array(settings['camera_matrix']['data'])
w = settings['camera_matrix']['cols']
h = settings['camera_matrix']['rows']
camera_matrix = buildCameraMatrix(w, h, raw_camera_matrix)

distortion_coefficients = np.array(settings['distortion_coefficients']['data'])
image_width = settings['image_width']
image_height = settings['image_height']

#Initialize camera
with PiCamera() as camera:
	div = .5
	w = int(1640 / div)
	h = int(1232 / div)
	camera.resolution = (w, h)#(760, 480)#(3280, 2464)
	camera.framerate = 5
	time.sleep(2)
	#Capture image
	myStreamFull = io.BytesIO()
	camera.capture(myStreamFull, format='jpeg')
	dataFull = np.frombuffer(myStreamFull.getvalue(), dtype=np.uint8)
	img = cv2.imdecode(dataFull, 1)


	if len(img.shape) == 3 and img.shape[2] == 3:
		mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		mono = img
	pattern_size = (9, 6)
	size = 0.108
	checkerboard_flags=cv2.CALIB_CB_FAST_CHECK
	print("Finding raw chessboard points...")
	(found, corners) = cv2.findChessboardCorners(mono, pattern_size, flags = cv2.CALIB_CB_ADAPTIVE_THRESH |
												cv2.CALIB_CB_NORMALIZE_IMAGE | checkerboard_flags)

	undistorted_img = cv2.undistort(mono, camera_matrix, distortion_coefficients)
	#Rescale the camera matrix
	# scale = int(3280/image_width)
	scale = 2
	for i in range(len(camera_matrix) - 1):
		camera_matrix[i] = camera_matrix[i] * scale

	pattern_points = calc_pattern_points(pattern_size, size)
	if found:
		print("Finding transform from object to camera")
		retval, rvec, tvec	= cv2.solvePnP(pattern_points, corners, camera_matrix, distortion_coefficients)

		print("Transforming object to camera")
		image_points, jacobian = cv2.projectPoints(pattern_points, rvec, tvec, camera_matrix, distortion_coefficients)

		print("Finding corners in undistored image...")
		(ufound, ucorners) = cv2.findChessboardCorners(undistorted_img, pattern_size, flags = cv2.CALIB_CB_ADAPTIVE_THRESH |
													cv2.CALIB_CB_NORMALIZE_IMAGE | checkerboard_flags)
		if ufound:
			print("Comparing...")
			if len(image_points) == len(ucorners):
				err = 0
				for i in range(len(image_points)):
					perrx = image_points[i][0][0] - ucorners[i][0][0]
					perry = image_points[i][0][1] - ucorners[i][0][1]
					cv2.circle(undistorted_img, tuple(image_points[i][0]), 1, (255,0,255), -1)
					cv2.circle(undistorted_img, tuple(ucorners[i][0]), 1, (0,255,0), -1)
					merr = math.sqrt(perrx*perrx + perry*perry)
					print("err: ", merr)
					err += merr
				err = err / len(image_points)
				print("Average radial error: ", err)
			else:
				print(len(image_points))
				print(len(corners))
		else:
			print("Failed undistored chessboard")
	else:
		print("FAIL")

	#Write images
	# cv2.imwrite("raw_test.jpg", img)
	cv2.imshow("und_test.jpg", undistorted_img)
	cv2.imwrite("und_test.jpg", undistorted_img)

	cv2.waitKey(0)