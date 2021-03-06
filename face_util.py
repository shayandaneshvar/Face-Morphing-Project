import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay


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
            cv2.circle(image, (x, y), 2, (0, 255, 255), 2)

    # detector output to numpy
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
        mean = np.mean(face_landmarks, axis=0)
        return np.subtract(face_landmarks, mean), mean

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
        face1_landmarks, _ = self.get_normalized_face_landmarks(
            image1)  # neutral
        face2_landmarks, _ = self.get_normalized_face_landmarks(image2)

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

        cv2.imshow('mean registered face landmarks', base)
        cv2.waitKey(0)
        cv2.destroyWindow('mean registered face landmarks')

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
        # print(sigma)

        for i in range(min(k, len(sigma))):
            for a in np.linspace(- sigma[i], sigma[i], 200):
                img = np.zeros((800, 800, 3), np.uint8)
                face = miu + np.matrix(a * U[:, i]).reshape((136, 1))
                for j in range(0, 136, 2):
                    cv2.circle(
                        img, (int(face[j, 0]) + 200, int(face[j + 1, 0]) + 200),
                        1, (255, 125, 125), 1)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow("face model")
                    return
                cv2.putText(img, f"sigma: {a}", (300, 500),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255))
                cv2.putText(img, f"i: {i}", (300, 600),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255))
                cv2.imshow("face model", img)
                cv2.waitKey(10)

        cv2.destroyWindow("face model")

    @staticmethod
    def triangulate(face):
        points = face.reshape(-1, 2)
        return Delaunay(points)

    @staticmethod
    def get_triangle_containing_point(triangles: Delaunay, x, y):
        triangle_index = triangles.find_simplex(np.array([x, y]))
        if triangle_index == -1:
            return None
        points_index = triangles.simplices[triangle_index]
        return triangles.points[points_index[0]], triangles.points[
            points_index[1]], triangles.points[points_index[2]]

    @staticmethod
    def get_triangle_index_containing_point(triangles: Delaunay, x, y):
        return triangles.find_simplex([[x, y]])

    @staticmethod
    def get_neutral_face_landmark_from_transformed_neutral_face(
            transformed_neutral_face, NL_landmarks, x, y):
        for i in range(0, transformed_neutral_face.shape[0], 2):
            if transformed_neutral_face[i, 0] == x and \
                    transformed_neutral_face[i + 1, 0] == y:
                return NL_landmarks[i, 0], NL_landmarks[i + 1, 0]
        return -1

    def open_camera(self, miu, U, neutral_image, triangles_only=False,
                    method="affine", approx=True):
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)
        NL, NL_Mean = self.get_normalized_face_landmarks(neutral_image)  # Nln
        NL_landmarks = NL + NL_Mean
        NL_landmarks = NL_landmarks.reshape((136, 1))
        M = None
        if method == "affine":
            print("Choosing Affine registration")
            M = FaceUtil._affine_register(miu.reshape((68, 2)), NL)
        else:
            print("Choosing Similarity registration")
            M = FaceUtil._similarity_register(miu.reshape((68, 2)), NL)

        NL = FaceUtil._apply_transform(NL, M)
        NL = NL.flatten().reshape((136, 1))

        while True:
            _, frame = cap.read()
            frame = frame[:, ::-1]  # mirror
            X, _ = self.get_normalized_face_landmarks(frame)
            img3 = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8) * 50
            M = FaceUtil._affine_register(miu.reshape((68, 2)), X)
            X = FaceUtil._apply_transform(X, M)

            X = X.flatten()
            X.resize((136, 1))
            # COLORING
            af = np.linalg.inv(U.T @ U) @ U.T @ (X - NL)
            transformed_neutral_face = NL + U @ af
            triangles: Delaunay = FaceUtil.triangulate(transformed_neutral_face)

            color_mappings = {}  # map: triangle -> color
            for i in range(triangles.simplices.shape[0]):
                pi1, pi2, pi3 = triangles.simplices[i]
                p1 = triangles.points[pi1]
                p2 = triangles.points[pi2]
                p3 = triangles.points[pi3]
                location_on_neutral = np.zeros((3, 2))

                location_on_neutral[0,
                :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                    transformed_neutral_face, NL_landmarks, p1[0], p1[1])
                location_on_neutral[
                1,
                :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                    transformed_neutral_face, NL_landmarks, p2[0], p2[1])
                location_on_neutral[
                2,
                :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                    transformed_neutral_face, NL_landmarks, p3[0], p3[1])
                point_location = location_on_neutral.mean(axis=0)  # simplex

                color_mappings[i] = FaceUtil.get_average_color(
                    neutral_image, int(point_location[0]),
                    int(point_location[1]))

            # X = miu + aU -> X - miu = aU -> a = (U.T @ U).inv @ U.T @ (X - miu)
            # a, _, __, ___ = np.linalg.lstsq(U, X - miu, rcond=None)
            # (A.T@A)-1 @ A.T @ b

            a = np.linalg.inv(U.T @ U) @ U.T @ (X - miu)
            face = miu + U @ a
            # print(a)

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
            if approx:
                self.color_image_with_neutral_face(color_mappings,
                                                   frame,
                                                   img3,
                                                   triangles, triangles_only)
            else:
                img3 = self.color_image_with_neutral_face_with_details(frame,
                                                                       img3,
                                                                       triangles,
                                                                       transformed_neutral_face,
                                                                       NL_landmarks,
                                                                       neutral_image)

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

    @staticmethod
    def color_image_with_neutral_face_with_details(frame, img3, triangles,
                                                   transformed_neutral_face,
                                                   NL_landmarks, neutral_image):
        s1 = frame.shape[0] / 1.8
        s2 = frame.shape[1] / 3
        neutral_image = neutral_image.copy()
        for i in range(triangles.simplices.shape[0]):
            pi1, pi2, pi3 = triangles.simplices[i]
            p1 = triangles.points[pi1]
            p2 = triangles.points[pi2]
            p3 = triangles.points[pi3]
            location_on_neutral = np.zeros((3, 2))

            location_on_neutral[0,
            :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                transformed_neutral_face, NL_landmarks, p1[0], p1[1])
            location_on_neutral[1,
            :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                transformed_neutral_face, NL_landmarks, p2[0], p2[1])
            location_on_neutral[2,
            :] = FaceUtil.get_neutral_face_landmark_from_transformed_neutral_face(
                transformed_neutral_face, NL_landmarks, p3[0], p3[1])

            pts1 = np.float32(location_on_neutral)
            pts2 = np.float32(
                [[p1[0] + s1, p1[1] + s2], [p2[0] + s1, p2[1] + s2],
                 [p3[0] + s1, p3[1] + s2]])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(neutral_image, M, (
                neutral_image.shape[1], neutral_image.shape[0]))
            # Create a rough mask on the triangle
            src_mask = np.zeros(
                (neutral_image.shape[0], neutral_image.shape[1], 3))
            poly = np.int32(pts2)
            cv2.fillPoly(src_mask, [poly], (255, 255, 255))
            poly_copied = np.multiply(src_mask, dst)
            mask = np.ones((neutral_image.shape[0], neutral_image.shape[1], 3))
            # assuming src1 and src2 are of same size

            cv2.fillPoly(mask, [poly], (0, 0, 0))
            img3 = np.multiply(mask, neutral_image.copy())
            img3 = np.add(poly_copied, img3)
        return cv2.resize(img3, (640, 480)) # fix this method idk what the f is this!

    @staticmethod
    def color_image_with_neutral_face(color_mappings, frame, img3,
                                      triangles, triangles_only=False):
        for i in range(triangles.simplices.shape[0]):
            pi1, pi2, pi3 = triangles.simplices[i]  # check index
            p1 = triangles.points[pi1]
            p2 = triangles.points[pi2]
            p3 = triangles.points[pi3]
            if triangles_only:
                cv2.line(img3, (
                    int(p1[0] + frame.shape[0] / 1.8),
                    int(p1[1] + frame.shape[1] / 3)),
                         (int(p2[0] + frame.shape[0] / 1.8),
                          int(p2[1] + frame.shape[1] / 3)), (255, 255, 255))
                cv2.line(img3, (
                    int(p1[0] + frame.shape[0] / 1.8),
                    int(p1[1] + frame.shape[1] / 3)),
                         (int(p3[0] + frame.shape[0] / 1.8),
                          int(p3[1] + frame.shape[1] / 3)), (255, 255, 255))
                cv2.line(img3, (
                    int(p3[0] + frame.shape[0] / 1.8),
                    int(p3[1] + frame.shape[1] / 3)),
                         (int(p2[0] + frame.shape[0] / 1.8),
                          int(p2[1] + frame.shape[1] / 3)), (255, 255, 255))
                continue
            contours = np.array([int(p1[0] + frame.shape[0] / 1.8),
                                 int(p1[1] + frame.shape[1] / 3),
                                 int(p3[0] + frame.shape[0] / 1.8),
                                 int(p3[1] + frame.shape[1] / 3),
                                 int(p2[0] + frame.shape[0] / 1.8),
                                 int(p2[1] + frame.shape[1] / 3)]) \
                .reshape(-1, 2)
            cv2.fillPoly(img3, [contours], (
                int(color_mappings[i][0]), int(color_mappings[i][1]),
                int(color_mappings[i][2])))

    @staticmethod
    def get_average_color(image, i, j, window_size: int = 1):
        if window_size % 2 == 0:
            window_size -= 1
        color = np.array([0, 0, 0])
        half_size = window_size // 2
        for x in range(i - half_size, i + half_size + 1):
            for y in range(j - half_size, j + half_size + 1):
                color += np.int32(image[y, x] / (window_size ** 2))
        return color
