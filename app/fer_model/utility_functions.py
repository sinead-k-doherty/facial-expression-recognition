import glob
import cv2
import dlib


# Resize, Crop and gray images
def prepare_image_dataset():
    detector = dlib.get_frontal_face_detector()
    extensions = ["tiff", "png"]
    datasets = ["test_images", "training_images"]
    for dataset in datasets:
        images = "/home/sinead/Documents/datasets/images/" + dataset + "/**/*."
        for ext in extensions:
            training_images = glob.glob(images + ext, recursive=True)
            for image in training_images:
                img = cv2.imread(image)
                dets = detector(img, 1)
                for i, d in enumerate(dets):
                    crop = img[d.top() : d.bottom(), d.left() : d.right()]
                    gray_image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(image, gray_image)


def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image, 0)
    dedected = detector(img, 1)
    if len(dedected) == 0:
        raise Exception("No face detected, please try again.")
    elif len(dedected) > 1:
        raise Exception("More than one face detected, please try again.")
    else:
        for i, d in enumerate(dedected):
            crop = img[d.top() : d.bottom(), d.left() : d.right()]
            cv2.imwrite(image, crop)
