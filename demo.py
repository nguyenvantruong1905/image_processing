## LOAD THU VIEN VA MODUL CAN THIET
import numpy as np
import cv2
import pytesseract

from PIL import Image
# if windows, install pytesseract and open comment bellow
# if not windows, read README.md to install pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\asus\AppData\Local\Tesseract-OCR\tesseract.exe'


# cv2.waitKey()

img = cv2.imread("test2.jpg")#load anh vao dau vao
cv2.imshow('Input', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# chuyen sang anh xam voi phuong thuc COLOR_BGR2GRAY
cv2.imshow('Input1', gray)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)# chuyen sang anh den trang dung phuong phap nguong thich ung cv2.adaptiveThreshold
cv2.imshow('Input2', thresh)
contours, h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,) #tim contours
largest_rectangle = [0, 0]
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
cv2.drawContours(img, [largest_rectangle[1]], 0, (0,255,0), 8)
cropped = img[y:y+h, x:x+w]
cv2.imshow('Input2', thresh)
cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),18)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening
cv2.imshow('Input4', invert)

cv2.waitKey()
cv2.destroyAllWindows()

