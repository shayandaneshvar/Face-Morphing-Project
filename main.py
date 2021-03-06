import glob

import cv2
import numpy as np

from face_util import FaceUtil

names = ["alireza", "shayan"]
transform = ["affine", "similarity"]


def read_images(person="alireza"):
    image_list = []
    files = glob.glob(f'images/{person}/*.*')
    files.sort()
    for filename in files:
        image = cv2.imread(filename)
        image_list.append(image)
    return image_list


# ------------ PHASE 1 -- face registration
def phase1(images, neutral_image, face_util, viz=True, method=transform[0],
           sample_number=6):
    face_util.show_faces_landmarks(images, sample_number)
    registered_faces = face_util.register_faces(images, neutral_image, method)
    if not viz:
        return registered_faces
    neutral_landmarks, _ = face_util.get_normalized_face_landmarks(
        neutral_image)
    temp_index = 0
    for registered_face in registered_faces:
        temp_index += 1
        if temp_index > sample_number:
            break
        base = np.zeros((500, 500, 3), np.uint8)
        cv2.putText(base, "Reds are the registered faces", (0, 80), 3, 0.65,
                    (255, 0, 255))
        cv2.putText(base, "The Greens are the neutral registered face",
                    (0, 120), 3, 0.65, (255, 0, 255))
        face_util.draw_face(base, registered_face, (0, 0, 255))
        face_util.draw_face(base, neutral_landmarks, (0, 255, 0))
        cv2.imshow('base', base)
        cv2.waitKey(0)
        cv2.destroyWindow('base')

    face_util.show_mean_face(registered_faces)
    return registered_faces


# ------------ PHASE 2 ------------- Face models + Animating principal modes
def phase2(face_util, registered_faces, k=16):
    face_util.animate_face(registered_faces, k)


# ------------ PHASE 3 -------------
def phase3(face_util, registered_faces, neutral_image, k=16,
           triangles_only=False, method="affine"):
    miu, U, sigma = face_util.find_pca(registered_faces, k)
    face_util.open_camera(miu, U, neutral_image, triangles_only, method)


if __name__ == '__main__':
    name = names[1]
    k = 14
    images = read_images(name)
    neutral_image = cv2.imread(f'images/{name}/1.jpg')
    face_util = FaceUtil("lib/shape_predictor_68_face_landmarks.dat")
    registered_faces = phase1(images,
                              neutral_image,
                              face_util,
                              viz=False,
                              method=transform[0], sample_number=0)
    # phase2(face_util, registered_faces, k=k)

    phase3(face_util, registered_faces, neutral_image, k=k,
           triangles_only=False, method=transform[0])  # and phase 4
