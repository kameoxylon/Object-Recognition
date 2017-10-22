## Yitzak Hernandez
## UCF
'''
images must be saved on a folder called "images" located on same place as project.py

This program uses template matching, histogram matching and sift matching to see if an image is similar to a group of images.
Images are scored between 1 and however many are in the images folder. with 1 being the highest.
Top four images are in no particular order.

Lines 35, 36, 40, 41, 45, 46 are commented out since all results are printed on screen. Just uncomment to see results.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def main():
	#Takes all of the images on the file specified folder
	mypath = "images/"
	fileNames = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
	imd = np.empty(len(fileNames), dtype=object)
	for n in range(0, len(fileNames)):
		imd[n] = cv2.imread( join(mypath,fileNames[n]) )

	print("Starting image matching process.")

	#Goes through each and every single image setting the first one as query, and all as database.
	for n in range(0, len(imd)):
		imq = imd[n].copy()

		templateTopFour, templateMatchingScores = templateMatching(imd, imq, n)
		#topFourImages(templateTopFour, fileNames, n, "Template Matching")
		#imageScorePlacing(templateMatchingScores, fileNames, n, "Template Matching")
		methodRanking(n, templateTopFour, "Template Matching", fileNames)

		histogramTopFour, histogramMatchingScores = histogramMatching(imd, imq, n)
		#topFourImages(histogramTopFour, fileNames, n, "Histogram Matching")
		#imageScorePlacing(histogramMatchingScores, fileNames, n, "Histogram Matching")
		methodRanking(n, histogramTopFour, "Histogram Matching", fileNames)

		siftTopFour, siftMatchingScores = SIFT(imd, imq, n)
		#topFourImages(siftTopFour, fileNames, n, "SIFT")
		#imageScorePlacing(siftMatchingScores, fileNames, n, "SIFT")
		methodRanking(n, siftTopFour, "SIFT", fileNames)

		print

	print("Done.")



def templateMatching(imd, imq, imageNumber):
	methods = ['cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED'] #[3, 1]

	#dict to hold template matching results.
	topFour = {}
	totalScores = {}


	for n in range(0, len(imd)):
		im = imd[n].copy()
		resArr = []

		for meth in methods:
			method = eval(meth)

			# Apply template Matching
			res = cv2.matchTemplate(im,imq,method)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			#Save each template score into an array. We take the inverse on TM_SQDIFF_NORMED
			if eval(meth) == 1:
				res = 1 - res
				resArr.append(res)
			else:
				resArr.append(res)
			
		pass
		#add each score of the array and take average
		score = np.sum(resArr) / len(resArr)
		totalScores[n] = score

		#save top 4 scores
		if len(topFour) < 4:
			topFour[n] = score
		else:
			topFour = topFourDictionary(topFour, n, score)

	return topFour, totalScores


def histogramMatching(imd, imq, imageNumber):
	#dict to hold final histogram matching results.
	topFour = {}
	totalScores = {}

	#calculate histogram of query image.
	histq = cv2.calcHist([imq], [0], None, [256], [0, 256])
	
	#calculate histogram of each database image.
	histd = np.empty(len(imd), dtype=object)
	for n in range(0, len(imd)):
		histd[n] = cv2.calcHist([imd[n]], [0], None, [256], [0, 256])

	#go through all the methods on each image and take average.
	for n in range(0, len(imd)):
		resArr = []
		resArr.append(cv2.compareHist(histq, histd[n], cv2.HISTCMP_CORREL))
		resArr.append(1 - cv2.compareHist(histq, histd[n], cv2.HISTCMP_BHATTACHARYYA))

		score = np.sum(resArr) / len(resArr)
		totalScores[n] = score

		#save top 4 scores
		if len(topFour) < 4:
			topFour[n] = score
		else:
			topFour = topFourDictionary(topFour, n, score)

	return topFour, totalScores



def SIFT(imd, imq, imageNumber):
	#dict to hold final SIFT matching results.
	topFour = {}
	totalScores = {}
	
	for z in range(0, len(imd)):
		img1 = imq.copy()
		img2 = imd[z].copy()

		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)

		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(des1,des2,k=2)

		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)

		score = len(good)
		totalScores[z] = score

		#save top 4 scores
		if len(topFour) < 4:
			topFour[z] = score
		else:
			topFour = topFourDictionary(topFour, z, score)

	return topFour, totalScores

#We save the location in the fileNames array (n), which holds the image name, and its score (score).
#If the new image score is lower we swap.
def topFourDictionary(topFour, n, score):
	for j, k in topFour.items():
				if k < score:
					del topFour[j]
					topFour[n] = score
					return topFour

	return topFour

def topFourImages(topFour, fileNames, image, method):
	print(fileNames[image] + " top matches in: " + method)

	for n, m in topFour.items():
		print fileNames[n]
	print

def imageScorePlacing(allScores, fileNames, image, method):
	print(fileNames[image] + " all scores in: " + method)
	count = len(allScores)

	for key, value in sorted(allScores.iteritems(), key = lambda (k, v): (v, k)):
		print count, fileNames[key]
		count = count - 1

	print

#Checks to see how each method did.
def methodRanking(image, topFour, method, fileNames):
	duck = [0, 1, 2, 3]
	chair = [4, 5, 6, 7]
	person = [8, 9, 10, 11]
	painting = [12, 13, 14, 15]
	picture = [16, 17, 18, 19]

	#This will keep track of the image and the score if they are in the correct place.
	count = 0
	for n, m in topFour.items():
		if (image in duck) & (n in duck):
			count = count + 1
		elif (image in chair) & (n in chair):
			count = count + 1
		elif (image in person) & (n in person):
			count = count + 1
		elif (image in painting) & (n in painting):
			count = count + 1
		elif (image in picture) & (n in picture):
			count = count + 1

	print("The score for " + fileNames[image]+ " using " + method + " = " + str(count))



if __name__ == "__main__":
	main()