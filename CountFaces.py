import cv2, numpy as np, dlib, os 

# Load the image
folder_path=r'C:\Users\Vijol Lobo\OneDrive\Desktop\VegaCam\DetectFaces' #Check the path of the folder
image_path = os.path.join(folder_path,"friends.jpg")#Check whether the image is in the same folder or not or rename the image or this file name
# Output path for the image with detected faces
output_path =os.path.join(folder_path,"faces_detected.jpg")

image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Iterator to count faces
i = 0
for face in faces:
    # Get the coordinates of faces
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)

    # Increment iterator for each detected face
    i += 1
    
    # Display the box and faces
    cv2.putText(image, f'Face {i}', (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(f"Face {i}: {face}")

# Save the new image with detected faces
cv2.imwrite(output_path, image)

# Display the resulting image
cv2.imshow('Detected Faces', image)
print(f"There are {i} faces in the input image")
print(f"Saved output image at: {output_path}")

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
