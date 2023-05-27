import cv2
from matplotlib import pyplot as plt
from pathlib import Path

# Read the image in color mode
image_path = "face_detection/test_image_3.jpg"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# Split the image into its color channels
b, g, r = cv2.split(img)
# Equalize the histogram for each color channel
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)
# Merge the equalized channels back together
img2 = cv2.merge([b_eq, g_eq, r_eq])
# Save the equalized image
path = Path(image_path)
cv2.imwrite(f"preprocessing/export/{path.name}", img2)
# Convert color for matplotlib (OpenCV uses BGR while matplotlib uses RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
