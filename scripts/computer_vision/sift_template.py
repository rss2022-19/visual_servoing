import cv2
import imutils
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	MIN_MATCH = 10
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	kp1, des1 = sift.detectAndCompute(template,None)
	kp2, des2 = sift.detectAndCompute(img,None)

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)
	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		h, w = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		########## YOUR CODE STARTS HERE ##########

		if False:
			img3 = cv2.drawMatches(template, kp1, img, kp2, good[:50], img) #, flags=2)
			image_print(img3)

		#Fix array pts to have 3rd dimension filled with ones
		pts = np.array([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
		i, j = pts.shape
		pts = np.hstack((pts,np.ones((i,1))))

		#Apply the transformation Matrix to the points
		pts2 = np.matmul(M,pts.T).T

		#Set #_mins and #_maxes
		x_min = min(pts2[:,0])
		y_min = min(pts2[:,1])
		x_max = max(pts2[:,0])
		y_max = max(pts2[:,1])

		########### YOUR CODE ENDS HERE ###########

		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print "[SIFT] not enough matches; matches: ", len(good)

		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))

def cd_template_matching(img, template):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	template_canny = cv2.Canny(template, 50, 200)

	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)

	# Get dimensions of template
	(img_height, img_width) = img_canny.shape[:2]

	# Keep track of best-fit match
	best_match = None

	#image_print(template)

	# Loop over different scales of image
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.

		method = "cv2.TM_CCOEFF_NORMED"
		is_sq = 0
		if "Q" in method:
			is_sq = 1
		alter = (1,-1)[is_sq]
		if "bounding_box" not in locals():
			bounding_box = ((0,0),(0,0))
		if "old_val" not in locals():
			if is_sq:
				old_val = 10**100
			else:
				old_val = 0

		res = cv2.matchTemplate(img_canny, resized_template, eval(method))
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		if is_sq:
			val = min_val
			loc = min_loc
		else:
			val = max_val
			loc = max_loc

		if val > alter*old_val:
			bounding_box = ((loc[0],loc[1]),
					(int(loc[0]+w*scale), int(loc[1]+h*scale)))
			old_val = val

		if False:
			print("Starting Report at scale "+str(scale)+" : ")
			print("min_loc", min_loc)
			print("max_loc", max_loc)
			print(bounding_box)
		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
		########### YOUR CODE ENDS HERE ###########
	print(bounding_box)
	print(old_val)
	#cv2.rectangle(img_canny,bounding_box[0], bounding_box[1], 255, 2)
	#image_print(img_canny)
	return bounding_box
