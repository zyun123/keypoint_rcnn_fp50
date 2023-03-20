import cv2
img = cv2.imread("/home/zy/Desktop/000001_cutout.png")
cv2.imshow("1",img[...,0])
cv2.imshow("2",img[...,1])
cv2.imshow("3",img[...,2])
# arr = img.split()
cv2.waitKey(0)
cv2.destroyAllWindows()

print("---------------------------")