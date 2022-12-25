import cv2
import tools.alpr_utils as utils

img1 = cv2.imread("004545.jpg")
img2 = cv2.imread("img/S__6201422.jpg")

r_img1 = utils.resize_image(img1, (416, 416))
r_img2 = utils.resize_image(img2, (416, 416))

# r_img1 = utils.resize_image(img1, (img2.shape[:2]))
cv2.imshow("r_img1", r_img1)
cv2.imshow("r_img2", r_img2)
print(img1.shape, r_img1.shape, img2.shape)
cv2.waitKey(0)