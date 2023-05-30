import cv2
import glob

def merge_rectangles():
    # Get all jpg and png files in the directory
    all_files = glob.glob('face_detection/export/*.[jp][np]g')

    for file_path in all_files:

        splitted_file_name = file_path.split('_-_')
        coordinates = splitted_file_name[1].split('.')[0]
        ext = splitted_file_name[1].split('.')[1]

        x,y,w,h = coordinates.split(',')
        top_left = (int(x), int(y))
        right_bottom = (int(w), int(h))

        original_file_path = splitted_file_name[0].replace('/export/', '/images/', 1)
        original_file_path_with_ext = f'{original_file_path}.{ext}'

        # read image file
        image = cv2.imread(original_file_path_with_ext)

        # Draw a bounding box on the face
        cv2.rectangle(image, top_left, right_bottom, (0, 255, 0), 2)

        # Put the emotion text above the bounding box
        cv2.putText(image, 'emotion_status_text', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imwrite(original_file_path_with_ext, image)
