## LOAD THU VIEN VA MODUL CAN THIET
import numpy as np
import cv2
# import pytesseract
from PIL import Image

img = cv2.imread("test7.jpg")#load anh vao dau vao
cv2.imshow('Input', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# chuyen sang anh xam voi phuong thuc COLOR_BGR2GRAY
cv2.imshow('Input1', gray)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)# chuyen sang anh den trang dung phuong phap nguong thich ung cv2.adaptiveThreshold
cv2.imshow('Input2', thresh)
contours, h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,) #tim contours
largest_rectangle = [0, 0, 0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.06*cv2.arcLength(cnt,True),True) # 0.06*cv2.arcLength(cnt,True) la khoang cach toi da tu duong bao den duong bao gan dung, ham tac dung lay sap si cac duong gap khuc nho thanh duong thang
    if len(approx)==4: 
        area = cv2.contourArea(cnt) #lay khu vuc trong duong vien
        # chon ra hinh chu nhat lon nhat
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[2])
image = img[y:y + h, x : x + w]
cv2.imshow('Input3', image)
# cv2.drawContours(img, [largest_rectangle[1]], 0, (0,255,0), 8)
cv2.waitKey()
cv2.destroyAllWindows()

