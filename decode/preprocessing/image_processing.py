import cv2

def histogram_equalization(image):

    """Applies histogram equalization to the original image"""

    # Split the image into its color channels
    b, g, r = cv2.split(image)
    # Equalize the histogram for each color channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    # Merge the equalized channels back together
    return cv2.merge([b_eq, g_eq, r_eq])
