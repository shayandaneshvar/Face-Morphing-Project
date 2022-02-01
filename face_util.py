import cv2
import dlib
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

    def get_face_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        if len(faces) == 0:
            return self.last_detected_face

        face = faces[0]
        face_landmarks = self.face_landmarks_detector(gray, face)
        face_landmarks = self.__shape_to_np(face_landmarks)
        self.last_detected_face = face_landmarks

        return face_landmarks

    def get_normalized_face_landmarks(self, image):
        face_landmarks = self.get_face_landmarks(image)

        return np.subtract(face_landmarks, np.mean(face_landmarks, axis=0))

    @staticmethod
    def _affine_register(face1_landmarks, face2_landmarks):
        coefficient_1 = (face2_landmarks[:, 0] * face2_landmarks[:, 0]).sum()
        coefficient_2 = (face2_landmarks[:, 1] * face2_landmarks[:, 0]).sum()
        coefficient_3 = (face2_landmarks[:, 1] * face2_landmarks[:, 1]).sum()

        coefficients_matrix = [[coefficient_1, coefficient_2, 0, 0],
                               [coefficient_2, coefficient_3, 0, 0],
                               [0, 0, coefficient_1, coefficient_2],
                               [0, 0, coefficient_2, coefficient_3]
                               ]

        equation_1_answer = (
                face1_landmarks[:, 0] * face2_landmarks[:, 0]).sum()
        equation_2_answer = (
                face1_landmarks[:, 0] * face2_landmarks[:, 1]).sum()
        equation_3_answer = (
                face2_landmarks[:, 0] * face1_landmarks[:, 1]).sum()
        equation_4_answer = (
                face1_landmarks[:, 1] * face2_landmarks[:, 1]).sum()

        answers = [equation_1_answer, equation_2_answer,
                   equation_3_answer, equation_4_answer]

        result = np.linalg.solve(coefficients_matrix, answers)

        return result.reshape((2, 2))

    @staticmethod
    def _similarity_register(face1_landmarks, face2_landmarks):
        coefficient = (face2_landmarks[:, 0] ** 2 +
                       face2_landmarks[:, 1] ** 2).sum()

        equation_1_answer = (face1_landmarks[:, 0] *
                             face2_landmarks[:, 0] +
                             face1_landmarks[:, 1] *
                             face2_landmarks[:, 1]).sum()

        equation_2_answer = (face1_landmarks[:, 1] *
                             face2_landmarks[:, 0] -
                             face1_landmarks[:, 0] *
                             face2_landmarks[:, 1]).sum()

        coefficients_matrix = [[coefficient, 0], [0, coefficient]]

        answers = [equation_1_answer, equation_2_answer]

        a, b = np.linalg.solve(coefficients_matrix, answers)

        return [[a, -b], [b, a]]

    @staticmethod
    def _apply_transform(image, transform_matrix):
        return image @ transform_matrix

    @staticmethod
    def draw_face(image, points, color):
        width = image.shape[0]
        height = image.shape[1]
        for (x, y) in points:
            cv2.circle(image, (int(x + width / 2), int(y + height / 2)), 1,
                       color, 2)

    def get_registered_face(self, image1, image2, method):
        face1_landmarks = self.get_normalized_face_landmarks(image1)  # neutral
        face2_landmarks = self.get_normalized_face_landmarks(image2)

        if method == "affine":
            M = FaceUtil._affine_register(face1_landmarks, face2_landmarks)
        elif method == "similarity":
            M = FaceUtil._similarity_register(face1_landmarks, face2_landmarks)
        else:
            raise Exception("invalid register method")

        return FaceUtil._apply_transform(face2_landmarks, M)

    def register_faces(self, faces, target_image, method):
        registered_faces = []

        for face in faces:
            registered_faces.append(
                self.get_registered_face(target_image, face, method))

        return np.array(registered_faces)

    def show_faces_landmarks(self, images, sample_numbers):
        for i in range(0, sample_numbers):
            image = images[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            for face in faces:
                face_landmarks = self.face_landmarks_detector(gray, face)
                self.__draw_face_land_marks(face_landmarks, image)

            cv2.imshow('face landmarks', image)
            cv2.waitKey(0)
            cv2.destroyWindow('face landmarks')

    @staticmethod
    def show_mean_face(faces):
        mean_face = np.mean(([face for face in faces]), axis=0)

        base = np.zeros((500, 500, 3), np.uint8)

        FaceUtil.draw_face(base, mean_face, (0, 255, 0))

        cv2.imshow('mean face landmarks', base)
        cv2.waitKey(0)
        cv2.destroyWindow('mean face landmarks')

    @staticmethod
    def _calc_miu(faces):
        return np.mean(([face.flatten() for face in faces]), axis=0).reshape(
            (136, 1))

    @staticmethod
    def find_pca(faces, k):
        miu = FaceUtil._calc_miu(faces)
        X = np.concatenate([face.flatten().reshape((136, 1))
                            for face in faces], axis=1)
        X -= miu
        U, sigma, _ = np.linalg.svd(X)  # _ V
        U = U[:, :k]
        sigma = sigma[:k]
        return miu, U, sigma

    @staticmethod
    def animate_face(faces, k):
        miu, U, sigma = FaceUtil.find_pca(faces, k)
        print(sigma)

        for i in range(min(k, len(sigma))):

            for a in np.linspace(- sigma[i], sigma[i], 200):
                img = np.zeros((800, 800, 3), np.uint8)
                face = miu + np.matrix(a * U[:, i]).reshape((136, 1))
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
                cv2.putText(img, f"sigma: {a}", (300, 500),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255))
                cv2.putText(img, f"i: {i}", (300, 600),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255))
                cv2.imshow("face model", img)
                cv2.waitKey(10)

    @staticmethod  # rect -> detect face
    def triangulate(face, rect, x, y):
        subdiv = cv2.Subdiv2D(rect)
        points = face.reshape(-1, 2)
        for point in points:
            p = (point[0] + x, point[1] + y)
            subdiv.insert(p)
        return subdiv.getTriangleList()

    def open_camera(self, miu, U, neutral_image):
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)

        while True:
            _, frame = cap.read()
            frame = frame[:, ::-1]  # mirror
            X = self.get_normalized_face_landmarks(frame)
            M = FaceUtil._affine_register(miu.reshape((68, 2)), X)
            X = FaceUtil._apply_transform(X, M)

            X = X.flatten()
            X.resize((136, 1))
            img3 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

            rect = [0, 0, frame.shape[0], frame.shape[1]]

            if len(rect) is 4:
                try:
                    triangles = FaceUtil.triangulate(X, rect,
                                                     frame.shape[0] / 1.8,
                                                     frame.shape[1] / 3)
                except Exception:
                    triangles = np.array([])
                    pass
                    # out of range ...

                for j in range(triangles.shape[0]):
                    x = triangles[j, 0] + triangles[j, 2] + triangles[j, 4]
                    y = triangles[j, 1] + triangles[j, 3] + triangles[j, 5]
                    x /= 3
                    y /= 3
                    M_inverse = np.linalg.inv(M)
                    location = M_inverse @ np.array([[x], [y]])
                    color = tuple([int(f) for f in
                                   neutral_image[int(location[0, 0])]
                                   [int(location[1, 0])]])

                    contours = np.array([np.int32(triangles[j, i])
                                         for i in range(6)]).reshape(-1, 2)

                    cv2.fillPoly(img3, [contours], color)

                    cv2.line(img3, (triangles[j, 0], triangles[j, 1]),
                             (triangles[j, 2], triangles[j, 3]),
                             (255, 255, 255))
                    cv2.line(img3, (triangles[j, 0], triangles[j, 1]),
                             (triangles[j, 4], triangles[j, 5]),
                             (255, 255, 255))
                    cv2.line(img3, (triangles[j, 4], triangles[j, 5]),
                             (triangles[j, 2], triangles[j, 3]),
                             (255, 255, 255))

            a, _, __, ___ = np.linalg.lstsq(U, X - miu, rcond=None)
            print(a)
            face = miu + U @ a

            img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
            img2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

            for j in range(0, 136, 2):
                cv2.putText(img, "The Other Guy", (50, 50), 3, 1, (0, 0, 200))
                cv2.circle(
                    img,
                    (int(face[j, 0] + frame.shape[0] / 1.8),
                     int(face[j + 1, 0] + frame.shape[1] / 3)),
                    2,
                    (200, 200, 50),
                    2,
                )
                cv2.putText(img2, "Original Face from webcam", (50, 50), 3, 1,
                            (0, 0, 200))
                cv2.circle(
                    img2,
                    (int(X[j, 0] + frame.shape[0] / 1.8),
                     int(X[j + 1, 0] + frame.shape[1] / 3)),
                    2,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(img3, "The Other Guy - triangulated", (50, 50), 3,
                            1, (0, 0, 200))

            img = np.hstack([img, frame])
            img2 = np.hstack([img2, img3])
            img = np.vstack([img, img2])
            scale = frame.shape[0] / frame.shape[1]
            img = cv2.resize(img, (768, int(768 * scale)))

            cv2.imshow("WebCam", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
