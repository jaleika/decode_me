import cv2
from face_detector import YoloDetector
import numpy as np
from PIL import Image

# Initialize the YOLOFace detector
yolo_face = YoloFaceDetection()

# Load the input image
image_path = "/Users/hayrikucuk/code/jaleika/decode_me/raw_data/Group5a/1000050894_5cc350a486_1369_9954378@N06.jpg"
image = cv2.imread(image_path)

# Perform face detection
faces = yolo_face.detect_faces(image)

# Draw bounding boxes around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
