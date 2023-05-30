from yolo5face.get_model import get_model
from pathlib import Path
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


def crop_faces(image, image_path: str) -> list[dict]:

    """Extracts faces from an image and returns the list of faces.
    Each face is represented by a dictionary, which contains the extracted face image, the name of the original image
    and the coordinates of the face on the original image.
    Args: image path
    """

    # Load the model
    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

    faces = []
    # Perform face detection
    boxes, _key_points = model(image)
    for face in boxes[0]:
        path = Path(image_path)
        faces.append(
            {
                "image": image[
                    face[1] : face[3],
                    face[0] : face[2],
                ],
                "image_name": path.name,
                "coordinates": face,
            }
        )
    return faces


def face_detection(image_path: str, should_equalize=True):

    """Reads the image from the path, then performs histogram equalization and extracts the faces"""

    image = cv2.imread(image_path)
    if should_equalize:
        equalized_image = histogram_equalization(image)
        return crop_faces(equalized_image, image_path)
    else:
        return crop_faces(image, image_path)


if __name__ == "__main__":
    print(face_detection("decode/ml_logic/face_detection/test_image_1.jpg"))
