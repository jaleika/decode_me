from yolo5face.get_model import get_model
from pathlib import Path
import cv2

def crop_faces(image_path: str = "face_detection/test_image_1.jpg"):

    """Extracts faces from an image and save them as separate images in "export" folder
    Args: image path
    """
    # Load the model
    model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)
    # Load an iamge from the path
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        # Predict the emotion for the cropped face
        emotion = load_dummy_model(f"face_detection/export/{path.stem}_-_{coordinates_string}.jpg")

        # Draw a bounding box on the face
        cv2.rectangle(image_copy, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

        # Put the emotion text above the bounding box
        cv2.putText(image_copy, emotion, (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Save the image with bounding boxes and emotion labels
    path = Path(image_path)
    cv2.imwrite(
        f"face_detection/export/{path.stem}_processed.jpg",
        image_copy,
    )

if __name__ == "__main__":
    crop_faces()
