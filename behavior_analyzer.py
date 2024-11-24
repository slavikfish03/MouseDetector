import numpy as np
import pandas
import os
import csv
import cv2

from ultralytics import YOLO

COUNT_FRAMES_IN_COMPOSITE_IMG = 21


class BehaviorAnalyzer:
    def __init__(self, height, width, path_to_behavior_weight_yolo, path_to_video):
        self.buffer_img = []
        self.buffer_behaviors = []
        self.info_behavior_of_mouse = None

        self.height = height
        self.width = width
        self.model = YOLO(path_to_behavior_weight_yolo)
        self.shift = COUNT_FRAMES_IN_COMPOSITE_IMG // 2

        self.output_name_csv = self.get_name_output_csv(path_to_video)
        self.init_csv_file()

    def init_csv_file(self):
        row = []
        for value in self.model.names.values():
            row.append(value)
        with open(f'{self.output_name_csv}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


    def create_composite_frame(self):
        index_current_img = COUNT_FRAMES_IN_COMPOSITE_IMG // 2
        current_frame = self.buffer_img[index_current_img]
        prev_frames = self.buffer_img[:index_current_img]
        next_frames = self.buffer_img[(index_current_img + 1):]

        current_red = current_frame[:, :, 2]

        avg_green = np.mean([frame[:, :, 1] for frame in prev_frames], axis=0)

        avg_blue = np.mean([frame[:, :, 0] for frame in next_frames], axis=0)

        composite_img = np.stack((avg_blue, avg_green, current_red), axis=2).astype(np.uint8)

        return composite_img

    def update_buffer(self, frame):

        if self.buffer_is_full():
            self.buffer_img.pop(0)
            self.buffer_img.append(frame)
        else:
            self.buffer_img.append(frame)

    def buffer_is_full(self):
        return len(self.buffer_img) >= COUNT_FRAMES_IN_COMPOSITE_IMG

    def analyze(self):
        composite_img = self.create_composite_frame()
        results = self.model(composite_img)
        for result in results:
            probs = result.probs
            probs_behavior_mouse = []
            for key in self.model.names.keys():
                probs_behavior_mouse.append(round(float(probs.data[key]), 3))

        self.export_to_csv(probs_behavior_mouse)

    def get_name_output_csv(self, path_to_video):
        output_filename = os.path.basename(path_to_video)
        name = output_filename.split('.')[0] + '_beh'
        return name

    def export_to_csv(self, probs_behavior_mouse):
        with open(f'{self.output_name_csv}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(probs_behavior_mouse)





