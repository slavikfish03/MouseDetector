import cv2
import numpy as np

THRESH = 90
MAXVAL_FOR_THRESHOLD = 255
COEFF_FOR_XY_ROI = 0.425
COEFF_FOR_WH_ROI = 0.4
MAX_CONTOUR_LENGTH = 100
DP_HOUGH = 1
MIN_DIST_HOUGH = 50
PARAM1_HOUGH = 30   #50 or 40
PARAM2_HOUGH = 40     #120 or 70
MIN_RADIUS_HOUGH = 200  #350
MAX_RADIUS_HOUGH = 500  #500


class AnalyticImageProcessor:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        self.x_center = None
        self.y_center = None
        self.radius_arena = None
        self.central_zone = None
        self.internal_zone = None
        self.middle_zone = None
        self.outer_zone = None

    def search_contour_keypoints_of_arena(self):
        x_roi = int(self.blurred_image.shape[1] * COEFF_FOR_XY_ROI)
        y_roi = int(self.blurred_image.shape[0] * COEFF_FOR_XY_ROI)
        w_roi = int(self.blurred_image.shape[0] * COEFF_FOR_WH_ROI)
        h_roi = int(self.blurred_image.shape[1] * COEFF_FOR_WH_ROI)

        roi = self.blurred_image[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]

        thresholded = cv2.threshold(roi, THRESH, MAXVAL_FOR_THRESHOLD, cv2.THRESH_BINARY)[1]
        all_contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_roi = []
        for contour in all_contours:
            contour_len = cv2.arcLength(contour, True)
            if contour_len < MAX_CONTOUR_LENGTH:
                contours_roi.append(contour)
        contour_keypoints_of_arena = []
        for contour in contours_roi:
            corrected_contour = contour + [x_roi, y_roi]
            contour_keypoints_of_arena.append(corrected_contour)
        return contour_keypoints_of_arena

    def search_contour_of_arena(self):
        detected_circle = cv2.HoughCircles(
            self.gray_image, cv2.HOUGH_GRADIENT, DP_HOUGH, MIN_DIST_HOUGH, param1=PARAM1_HOUGH,
            param2=PARAM2_HOUGH, minRadius=MIN_RADIUS_HOUGH, maxRadius=MAX_RADIUS_HOUGH)

        x_center, y_center = int(detected_circle[0, 0][0]), int(detected_circle[0, 0][1])
        radius = int(detected_circle[0, 0][2])

        self.radius_arena = radius
        return x_center, y_center, radius

    def search_center(self):
        contour_keypoints_of_arena = self.search_contour_keypoints_of_arena()
        x_center_contour, y_center_contour, radius_contour = self.search_contour_of_arena()
        min_dist = float('inf')
        contour_center = None
        for contour in contour_keypoints_of_arena:
            distance = cv2.pointPolygonTest(contour, (x_center_contour, y_center_contour), True)
            if abs(distance) < min_dist:
                min_dist = abs(distance)
                contour_center = contour

        M = cv2.moments(contour_center)
        x_center = int(M["m10"] / M["m00"])
        y_center = int(M["m01"] / M["m00"])

        self.x_center = x_center
        self.y_center = y_center
        return x_center, y_center, contour_center

    def search_central_zone(self):
        x_center, y_center, contour_center = self.search_center()
        contour_keypoints_of_arena = self.search_contour_keypoints_of_arena()

        nearest_contour = None
        min_dist = float('inf')
        for contour in contour_keypoints_of_arena:
            if np.array_equal(contour_center, contour):
                continue
            distance = cv2.pointPolygonTest(contour, (x_center, y_center), True)
            if abs(distance) < min_dist:
                min_dist = int(abs(distance))
                nearest_contour = contour

        M = cv2.moments(nearest_contour)
        x_radius = int(M["m10"] / M["m00"])
        y_radius = int(M["m01"] / M["m00"])

        dist = int(np.sqrt((x_center - x_radius) ** 2 + (y_center - y_radius) ** 2))

        self.central_zone = (0, dist)
        return x_center, y_center, dist

    def search_internal_zone(self):
        radius = round((self.radius_arena - self.central_zone[1]) / 3 + self.central_zone[1])
        self.internal_zone = (self.central_zone[1], radius)

    def search_middle_zone(self):
        radius = round(2 * (self.radius_arena - self.central_zone[1]) / 3 + self.central_zone[1])
        self.middle_zone = (self.internal_zone[1], radius)

    def search_outer_zone(self):
        self.outer_zone = (self.middle_zone[1], self.radius_arena)

    def search_zones(self):
        self.search_central_zone()
        self.search_internal_zone()
        self.search_middle_zone()
        self.search_outer_zone()

    def draw_zones(self, image: np.ndarray) -> np.ndarray:
        image = cv2.circle(image, (self.x_center, self.y_center), 2, (0, 0, 255), -1)
        image = cv2.circle(image, (self.x_center, self.y_center), self.central_zone[1], (0, 0, 255), 2)
        image = cv2.circle(image, (self.x_center, self.y_center), self.internal_zone[1], (0, 255, 0), 2)
        image = cv2.circle(image, (self.x_center, self.y_center), self.middle_zone[1], (255, 0, 0), 2)
        image = cv2.circle(image, (self.x_center, self.y_center), self.outer_zone[1], (255, 0, 255), 2)
        return image

    def show(self):
        self.search_zones()
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()