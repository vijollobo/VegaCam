import cv2, os 
# pre-trained model used by OpenCV for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

folder_path=r'C:\Users\Vijol Lobo\OneDrive\Desktop\VegaCam\DetectFaces' #Check the path of the folder
image_path = os.path.join(folder_path,"friends.jpg")#Check whether the image is in the same folder or not or rename the image or this file name
# Output path for the image with detected faces
output_path =os.path.join(folder_path,"faces_counted.jpg")

# Load an image
image = cv2.imread(image_path)
# image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)
cv2.imshow('Faces Detected', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
