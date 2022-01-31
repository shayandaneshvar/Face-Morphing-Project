import dlib
import cv2
from matplotlib.pyplot import axis
import numpy as np

class FaceUtil:
  def __init__(self, dlib_predictor_path):
    self.face_detector = dlib.get_frontal_face_detector()
    self.face_landmarks_detector = dlib.shape_predictor(dlib_predictor_path)
    self.last_detected_face = np.random.randint(0, 100, (68, 2), np.uint8)
  
  def __pre_process_image(self, image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  def __draw_face_land_marks(self, face_landmarks, image):
    for n in range(0, 68):
      x = face_landmarks.part(n).x
      y = face_landmarks.part(n).y
      cv2.circle(image, (x, y), 1, (0, 255, 255), 1)

  def __shape_to_np(self, shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

  def get_facelandmarks(self, image):
    gray = self.__pre_process_image(image)
    faces = self.face_detector(gray)

    if len(faces) == 0:
      return self.last_detected_face
      
    face = faces[0]
    face_landmarks = self.face_landmarks_detector(gray, face)
    face_landmarks = self.__shape_to_np(face_landmarks)
    self.last_detected_face = face_landmarks

    return face_landmarks

  def get_normalized_facelandmarks(self, image):
      face_landmarks = self.get_facelandmarks(image)

      return np.subtract(face_landmarks, np.mean(face_landmarks, axis=0))

  def __affine_register(self, face1_landmarks, face2_landmarks):
      coefficient_1 = (face2_landmarks[:, 0] * face2_landmarks[:, 0]).sum()
      coefficient_2 = (face2_landmarks[:, 1] * face2_landmarks[:, 0]).sum()
      coefficient_3 = (face2_landmarks[:, 1] * face2_landmarks[:, 1]).sum()

      coefficients_matrix = [[coefficient_1, coefficient_2, 0, 0],
                      [coefficient_2, coefficient_3, 0, 0],
                      [0, 0, coefficient_1, coefficient_2],
                      [0, 0, coefficient_2, coefficient_3]
                      ]

      equation_1_answer = (face1_landmarks[:, 0] * face2_landmarks[:, 0]).sum()
      equation_2_answer = (face1_landmarks[:, 0] * face2_landmarks[:, 1]).sum()
      equation_3_answer = (face2_landmarks[:, 0] * face1_landmarks[:, 1]).sum()
      equation_4_answer = (face1_landmarks[:, 1] * face2_landmarks[:, 1]).sum()

      answers = [equation_1_answer, equation_2_answer,
                equation_3_answer, equation_4_answer]

      result = np.linalg.solve(coefficients_matrix, answers)

      return result.reshape((2,2))

  def __similarity_register(self, face1_landmarks, face2_landmarks):
    coefficient = (face2_landmarks[:,0] ** 2 + face2_landmarks[:,1] ** 2).sum()
    equation_1_answer = (face1_landmarks[:,0] * face2_landmarks[:,0] + face1_landmarks[:,1] * face2_landmarks[:,1]).sum()
    equation_2_answer = (face1_landmarks[:,1] * face2_landmarks[:,0] - face1_landmarks[:,0] * face2_landmarks[:,1]).sum()
    coefficients_matrix = [[coefficient, 0], [0, coefficient]]
    answers = [equation_1_answer, equation_2_answer]

    a, b = np.linalg.solve(coefficients_matrix, answers)

    return [[a, -b], [b, a]]

  def __apply_transform(self, image, transformMatrix):
    return image @ transformMatrix

  def draw_face(self, image, points, color):
    width = image.shape[0]
    height = image.shape[0]
    for (x, y) in points:
        cv2.circle(image, (int(x + width / 2),
                   int(y + height / 2)), 1, color, 1)

  def get_registered_face(self, image1, image2, method):
    face1_landmarks = self.get_normalized_facelandmarks(image1)
    face2_landmarks = self.get_normalized_facelandmarks(image2)

    if method == "affine":
      M = self.__affine_register(face1_landmarks, face2_landmarks)
    elif method == "similarity":
      M = self.__similarity_register(face1_landmarks, face2_landmarks)
    else:
      raise Exception("invalid register method")

    return self.__apply_transform(face2_landmarks, M)

  def register_faces(self, faces, target_image, method):
    registered_faces = []

    for face in faces:
      registered_faces.append(self.get_registered_face(target_image, face, method))

    return np.array(registered_faces)
            
  def show_faces_landmarks(self, images):
    for image in images:
      gray = self.__pre_process_image(image)
      faces = self.face_detector(gray)

      for face in faces:
        face_landmarks = self.face_landmarks_detector(gray, face)
        self.__draw_face_land_marks(face_landmarks, image)

      cv2.imshow('face landmarks', image)
      cv2.waitKey(0)
      cv2.destroyWindow('face landmarks')

  def show_mean_face(self, faces):
    mean_face = np.mean(([face for face in faces]), axis=0)

    base = np.zeros((400, 400, 3), np.uint8)

    self.draw_face(base, mean_face, (0, 255, 0))

    cv2.imshow('mean face landmarks', base)
    cv2.waitKey(0)
    cv2.destroyWindow('mean face landmarks')

  def __calc_miu(self, faces):
    return np.mean(([face.flatten() for face in faces]), axis=0).reshape((136, 1))

  def find_pca(self, faces, K):
    miu = self.__calc_miu(faces)
    X = np.concatenate([face.flatten().reshape((136, 1)) for face in faces], axis=1)
    X -= miu
    U, sigma, _ = np.linalg.svd(X)
    U = U[:, :K]
    sigma = sigma[:K]
    return miu, U, sigma

  def animate_face(self, faces, K):
    miu, U, sigma = self.find_pca(faces, K)
    print(sigma)

    for i in range(min(K, len(sigma))):
      for a in np.arange(-abs(sigma[i] * 2), abs(sigma[i] * 2), 1):
        img = np.zeros((1024, 1024, 3), np.uint8)
        face = miu + a * U[:, i]
        for j in range(0, 136, 2):
            cv2.circle(
                img,
                (int(face[j, 0]) + 200, int(face[j + 1, 0]) + 200),
                1,
                (255, 125, 125),
                1,
            )
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
        cv2.putText(img, str(a), (500, 500), 2, 4, (0, 0, 255))
        cv2.imshow("face model", img)
        cv2.waitKey(10)
  
  def open_camera(self, miu, U):
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    while True:
        _, frame = cap.read()
        X = self.get_normalized_facelandmarks(frame)
        M = self.__affine_register(miu.reshape((68, 2)), X)
        X = self.__apply_transform(X, M)

        X = X.flatten()
        X.resize((136, 1))

        a, _, __, ___ = np.linalg.lstsq(U, X - miu, rcond=None)

        face = miu + U @ a

        img = np.zeros((1024, 1024, 3), np.uint8)
        for j in range(0, 136, 2):
            cv2.circle(
                img,
                (2 * int(face[j, 0]) + 200, 2 * int(face[j + 1, 0]) + 200),
                1,
                (255, 0, 0),
                1,
            )
            cv2.circle(
                img,
                (2 * int(X[j, 0]) + 200, 2 * int(X[j + 1, 0]) + 200),
                1,
                (0, 255, 255),
                1,
            )
        cv2.imshow("WebCam", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

