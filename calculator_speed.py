import numpy as np

LENGTH_MOUSE_IN_METERS = 0.08
RADUIS_ARENA_IN_METERS = 0.32


class CalculatorSpeed:
    def __init__(self):
        self.info_mouse_prev_frame = None
        self.info_mouse_current_frame = None
        self.time_prev_frame = None
        self.time_current_frame = None

    def update(self, info_mouse_current_frame, radius_arena, time_current_frame):
        if self.info_mouse_prev_frame is None:
            self.info_mouse_prev_frame = info_mouse_current_frame
            self.time_prev_frame = time_current_frame
            return 0.0
        else:
            speed = self.calculate_speed(info_mouse_current_frame, radius_arena, time_current_frame)
            self.info_mouse_prev_frame = info_mouse_current_frame.copy()
            self.time_prev_frame = time_current_frame
            return speed

    def calculate_speed(self, info_mouse_current_frame, radius_arena, time_current_frame):
        center_mouse_prev_frame = self.calculate_center_of_mouse(self.info_mouse_prev_frame)
        center_mouse_current_frame = self.calculate_center_of_mouse(info_mouse_current_frame)
        distance = np.sqrt((center_mouse_prev_frame[0] - center_mouse_current_frame[0])**2 +
                           (center_mouse_prev_frame[1] - center_mouse_current_frame[1])**2)
        time_diff = time_current_frame - self.time_prev_frame
        speed_px_per_seconds = distance / time_diff
        speed = self.convert_speed_to_meters_per_seconds(speed_px_per_seconds, info_mouse_current_frame, radius_arena)
        return round(speed, 3)

    def calculate_center_of_mouse(self, info_mouse) -> tuple[int, int]:
        center_head = (info_mouse['point_nose'][0] // 2 + info_mouse['point_near'][0] // 2,
                       info_mouse['point_nose'][1] // 2 + info_mouse['point_near'][1] // 2)
        center_body = (info_mouse['point_near'][0] // 2 + info_mouse['point_tail'][0] // 2,
                       info_mouse['point_near'][1] // 2 + info_mouse['point_tail'][1] // 2)

        center_mouse = (center_head[0] // 2 + center_body[0] // 2, center_head[1] // 2 + center_body[1] // 2)
        return center_mouse

    def calculate_length_mouse_in_px(self, info_mouse):
        vector_body = (info_mouse['point_near'][0] - info_mouse['point_tail'][0],
                       info_mouse['point_near'][1] - info_mouse['point_tail'][1])

        vector_head = (info_mouse['point_near'][0] - info_mouse['point_nose'][0],
                       info_mouse['point_near'][1] - info_mouse['point_nose'][1])

        length_vector_body = np.linalg.norm(vector_body)
        length_vector_head = np.linalg.norm(vector_head)

        return length_vector_body + length_vector_head

    def convert_speed_to_meters_per_seconds(self, speed, info_mouse_current_frame, radius_arena):
        one_px_in_meters = RADUIS_ARENA_IN_METERS / radius_arena
        length_mouse_in_px = self.calculate_length_mouse_in_px(info_mouse_current_frame)
        length_mouse_in_meters = length_mouse_in_px * one_px_in_meters
        speed = (length_mouse_in_meters / length_mouse_in_px) * speed
        return speed
