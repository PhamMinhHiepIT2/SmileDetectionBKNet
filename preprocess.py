import os
import cv2

from mtcnn.mtcnn import MTCNN
from pathlib import Path


detector = MTCNN()


def face_detection(image_path: str):
    """
    Detect face position in raw image

    Args:
        image_path (str): path to image

    Returns:
        Detected face position (x, y, w, h)
    """
    img = cv2.imread(image_path)
    detected_face = detector.detect_faces(img)
    if not detected_face:
        print("Cannot detect face")

    face_position = detected_face[0].get('box')
    x = face_position[0]
    y = face_position[1]
    w = face_position[2]
    h = face_position[3]
    return x, y, w, h


def preprocess_data(image_folder, des_image_folder):
    """
    Crop face from all images in folder and resize to 28x28 grayscale images

    Args:
        image_folder (str): image raw folder
        des_image_folder (str): processed image folder
    """
    Path(image_folder).mkdir(parents=True, exist_ok=True)
    images_path = [os.path.join(image_folder, file)
                   for file in os.listdir(image_folder)]
    for path in images_path:
        x, y, w, h = face_detection(path)
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        new_image_path = os.path.join(des_image_folder, os.path.basename(path))
        cv2.imwrite(new_image_path, img)


if __name__ == "__main__":
    data_path = '/Users/admin/Development/Dev/ComputerVision/SmileDetection/dataset/smile'
    processed_folder = '/Users/admin/Development/Dev/ComputerVision/SmileDetection/dataset/SMILEsmileD/SMILEs/positives/positives7'
    preprocess_data(data_path, processed_folder)
