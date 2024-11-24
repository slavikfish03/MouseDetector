import math

import numpy as np

EPS_FOR_LENGTH_VECTORS = 0.1

class Calculator:
    def __init__(self, info_mouse: dict, info_arena: dict, time_frame: tuple[str, str]):
        self.info_mouse = info_mouse
        self.info_arena = info_arena
        self.time_frame = time_frame

    def calculate(self):
        row_data = [f'{self.time_frame[0]}:{self.time_frame[1]}']
        xy = self.calculate_xy_mouse()
        row_data.append(xy[0])
        row_data.append(xy[1])

        zoning = self.calculate_zone_mouse()
        for zone in zoning:
            row_data.append(zone)

        angle = self.calculate_angle_head_body()
        row_data.append(round(angle))

        return row_data

    def calculate_center_of_mouse(self) -> tuple[int, int]:
        center_head = (self.info_mouse['point_nose'][0] // 2 + self.info_mouse['point_near'][0] // 2,
                       self.info_mouse['point_nose'][1] // 2 + self.info_mouse['point_near'][1] // 2)
        center_body = (self.info_mouse['point_near'][0] // 2 + self.info_mouse['point_tail'][0] // 2,
                       self.info_mouse['point_near'][1] // 2 + self.info_mouse['point_tail'][1] // 2)

        center_mouse = (center_head[0] // 2 + center_body[0] // 2, center_head[1] // 2 + center_body[1] // 2)
        return center_mouse

    def change_coordinate_system(self, x_old: int, y_old: int) -> tuple[int, int]:
        x_center, y_center = (self.info_arena['x_center'], self.info_arena['y_center'])
        new_x = x_old - x_center
        new_y = y_center - y_old
        return new_x, new_y

    def calculate_xy_mouse(self) -> tuple[int, int]:
        x_mouse, y_mouse = self.calculate_center_of_mouse()
        new_x_mouse, new_y_mouse = self.change_coordinate_system(x_mouse, y_mouse)
        return new_x_mouse, new_y_mouse

    def calculate_zone_mouse(self) -> tuple[bool, ...]:
        zoning_dict = {
            self.info_arena['central_zone']: 0,
            self.info_arena['internal_zone']: 0,
            self.info_arena['middle_zone']: 0,
            self.info_arena['outer_zone']: 0
        }
        for keypoint in self.info_mouse.values():
            new_x, new_y = self.change_coordinate_system(keypoint[0], keypoint[1])
            distance = np.sqrt(new_x**2 +
                               new_y**2)
            for (r_min, r_max), value in zoning_dict.items():
                if r_min < distance < r_max:
                    zoning_dict[(r_min, r_max)] += 1
        values = list(zoning_dict.values())
        max_index = values.index(max(values))
        result = [False] * len(values)
        result[max_index] = True

        return tuple(result)

    def calculate_angle_head_body(self):
        vector_body = (self.info_mouse['point_near'][0] - self.info_mouse['point_tail'][0],
                       self.info_mouse['point_near'][1] - self.info_mouse['point_tail'][1])

        # vector_head = (self.info_mouse['point_near'][0] - self.info_mouse['point_nose'][0],
        #                self.info_mouse['point_near'][1] - self.info_mouse['point_nose'][1])
        vector_head = (self.info_mouse['point_nose'][0] - self.info_mouse['point_near'][0],
                       self.info_mouse['point_nose'][1] - self.info_mouse['point_near'][1])

        length_vector_body = np.linalg.norm(vector_body)
        length_vector_head = np.linalg.norm(vector_head)

        print("Длины векторов (тело, голова): ", length_vector_body, length_vector_head)

        if length_vector_body < EPS_FOR_LENGTH_VECTORS or length_vector_head < EPS_FOR_LENGTH_VECTORS:
            angle = 0
        else:
            dot = np.dot(vector_body, vector_head)
            det = - vector_body[0] * vector_head[1] + vector_body[1] * vector_head[0]
            angle = np.arctan2(det, dot)
            angle = np.degrees(angle)

            if angle < 0:
                angle += 360

        print('Угол: ', angle)

        return angle




