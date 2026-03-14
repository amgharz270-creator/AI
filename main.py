import cv2


img1 = cv2.imread("images/image1.jpg")
img2 = cv2.imread("images/image2.jpg")


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()


kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Keypoints image1:", len(kp1))
print("Keypoints image2:", len(kp2))


bf = cv2.BFMatcher()


matches = bf.knnMatch(des1, des2, k=2)


good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print("Good matches:", len(good))


result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)


cv2.imshow("SIFT Matching", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
