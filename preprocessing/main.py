import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

image_path = "face_detection/test_image_3.jpg"
img = cv2.imread(image_path, 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# plt.plot(cdf_normalized, color="b")
# plt.hist(img.flatten(), 256, [0, 256], color="r")
# plt.xlim([0, 256])
# plt.legend(("cdf", "histogram"), loc="upper left")
# plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype("uint8")

img2 = cdf[img]
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# path = Path(image_path)
# cv2.imwrite(
#     f"preprocessing/export/{path}.jpg",
#     img2,
# )

imgplot = plt.imshow(img2)
plt.show()
