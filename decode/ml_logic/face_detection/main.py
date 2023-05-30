from yolo5face.get_model import get_model
from pathlib import Path
import cv2


def histogram_equalization(image):

    """Applies histogram equalisation to the original image"""

    # Split the image into its color channels
    b, g, r = cv2.split(image)
    # Equalize the histogram for each color channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    # Merge the equalized channels back together
    return cv2.merge([b_eq, g_eq, r_eq])


def crop_faces(image, image_path):

    """Extracts faces from an image and save them as separate images in "export" folder
    Args: image path
    """

    # Load the model
    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

    # Perform face detection
    faces, _key_points = model(image)
    for index, face in enumerate(faces[0]):
        cropped_image = image[
            face[1] : face[3],
            face[0] : face[2],
        ]
        path = Path(image_path)
        coordinates_string = ",".join(map(str, face))
        cv2.imwrite(
            f"face_detection/export/{path.stem}_-_{coordinates_string}.jpg",
            cropped_image,
        )


def face_detection(
    image_path: str = "face_detection/test_image_1.jpg", should_equalize=True
):

    """Reads the image from the path, then performs histogram equalization and extracts the faces"""

    image = cv2.imread(image_path)
    if should_equalize:
        equalized_image = histogram_equalization(image)
        crop_faces(equalized_image, image_path)
    else:
        crop_faces(image, image_path)


if __name__ == "__main__":
    main()
