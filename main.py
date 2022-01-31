from face_util import FaceUtil
import glob
import cv2
import numpy as np

def read_images():
  image_list = []
  for filename in glob.glob('images/alireza/*.jpg'):
    image = cv2.imread(filename)
    image_list.append(image)
  return image_list

images = read_images()
neutral_image = cv2.imread('images/alireza/neutral.jpg')

face_util = FaceUtil("lib/shape_predictor_68_face_landmarks.dat")

# ------------ PHASE 1 -------------

# face_util.show_faces_landmarks(images)

registered_faces = face_util.register_faces(images, neutral_image, "affine")
# neutral_landmarks = face_util.get_normalized_facelandmarks(neutral_image)

# for image in registered_faces:
#   base = np.zeros((400, 400, 3), np.uint8)

#   face_util.draw_face(base, image, (0, 0, 255))
#   face_util.draw_face(base, neutral_landmarks, (0, 255, 0))

#   cv2.imshow('base', base)
#   cv2.waitKey(0)
#   cv2.destroyWindow('base')

# face_util.show_mean_face(registered_faces)

# ------------ PHASE 2 -------------

K = 16
# face_util.animate_face(registered_faces, K)


# ------------ PHASE 3 -------------

miu, U, sigma = face_util.find_pca(registered_faces, K)
face_util.open_camera(miu, U)

