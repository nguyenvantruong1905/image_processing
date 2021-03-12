import cv2
import imutils
import numpy as np
import pytesseract as tess
from PIL import Image
  
im = cv2.imread("test6.jpg")#load anh

cv2.imshow('Input', im)
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#chuyen sang anh xam
cv2.imshow('image1', im_gray)
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)#filter khác với các filter khác là nó kết hợp cả domain filters(linear filter) và range filter(gaussian filter). Mục đích là giảm noise và tăng edge(làm egde thêm sắc nhọn edges sharp)
cv2.imshow('Input image2',noise_removal)
equal_histogram = cv2.equalizeHist(noise_removal) # làm cho ảnh ko quá sáng hoặc tối
cv2.imshow('image3', equal_histogram)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) #
cv2.imshow('image4', kernel)
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)#Morphogoly open ( open là erosion sau đó dilation) mục đích là giảm egde nhiễu , egde thật thêm sắc nhọn bằng cv2.morphologyEx sử dụng kerel 5x5
cv2.imshow('image5', morph_image)
sub_morp_image = cv2.subtract(equal_histogram,morph_image) #Xóa phông(background) không cần thiết
cv2.imshow('image6', sub_morp_image)
blur = cv2.GaussianBlur(sub_morp_image,(5,5),0)
ret,thresh_image = cv2.threshold(blur,0,255,cv2.THRESH_OTSU) #đưa ảnh về trắng đen tách biệt background và region interesting
cv2.imshow('image7', thresh_image)
canny_image = cv2.Canny(thresh_image,250,255) # Sử dụng thuật toán Canny để nhận biết egde 
cv2.imshow('Image8', canny_image)
kernel1 = np.ones((3,3), np.uint8) #Cuối cùng dilate để tăng sharp cho egde
cv2.imshow('image9', kernel1)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.imshow('image10', dilated_image)
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #tìm contour



contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] #Lọc contour theo area chỉ lấy 10 contour có giá trị lớn nhất
largest_rectangle = [0, 0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.06*cv2.arcLength(cnt,True),True)
    if len(approx)==4: 
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
# cv2.drawContours(im, contours, -1, (0, 255, 0), 2) #ve contours

# Xac dinh cac canh cua contour
x,y,w,h = cv2.boundingRect(largest_rectangle[1])
image = im[y:y + h, x : x + w]
cv2.drawContours(im, [largest_rectangle[1]], 0, (0,255,0), 8)
cropped = im[y:y+h, x:x+w]
cv2.drawContours(im,[largest_rectangle[1]],0,(255,255,255),18)
cv2.imshow('cro', cropped)
cv2.imshow('contour', im)
cv2.waitKey()
cv2.destroyAllWindows()

gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # chuyen ve anh xam
blur = cv2.GaussianBlur(gray, (3,3), 0)# lam mo anh
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]#chuyen anh den trang
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening#dao nguoc anh 
cv2.imshow('cro1', invert)
text = tess.image_to_string(invert,lang="eng")#chuyen anh bien so sang text
plate = text.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})#loai bo ki tu dac biet
print (plate)
f = open('demo_file.txt', 'a')#ghi text vao file

f.write(plate)
f.close()
