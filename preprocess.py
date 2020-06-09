import cv2
import imutils
import os
import numpy as np

# Load HandPose Model
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22

POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Function to Convert Normal Photos to HandPose
def draw_keypoints(img):
    img1 = np.zeros((img.shape[1],img.shape[1],3), np.uint8)

    aspect_ratio = img.shape[1]/img.shape[0]
    threshold = 0.1

    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    inpBlob = cv2.dnn.blobFromImage(img, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB = False, crop = False)

    net.setInput(inpBlob)

    output = net.forward()

    points = []
    points_X = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            points.append((int(point[0]), int(point[1])))
            points_X.append([int(point[0]), int(point[1])])
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img1, points[partA], points[partB], (255, 255, 255), 2)
            cv2.line(img, points[partA], points[partB], (0, 255, 255), 2)
    return img1, img

# Uncomment this section only during preprocessing.
'''    
# Process all images in your Dataset
path = "test_images/data"

dirs = os.listdir(path)

for i in dirs:
    print("Reading "+i)
    path1 = path + "/" + i
    for j in os.listdir(path1):
        path2 = path1 + "/" + j
        img = cv2.imread(path2)
        skel,im = draw_keypoints(img)
        try:
            cv2.imwrite('data/train/'+i+'/'+j, skel)
        except:
            os.mkdir('data/train/'+i)
            cv2.imwrite('data/train/' + i + '/' + j, skel)
        print(j)
    print(i+" done")

'''
