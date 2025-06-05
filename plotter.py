import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from calculator_speed import RADUIS_ARENA_IN_METERS

ZONES_ARENA = ["Central zone", "Internal zone", "Middle zone", "Outer zone"]
BEHAVIORS = ['groom', 'run', 'sit']
BEHAVIORS_PALETTE = {'groom': 'blue', 'run': 'green', 'sit': 'red'}
STEP_TIME_FOR_ETHOGRAM = 30

class Plotter:
    def __init__(self, path_to_beh, radius_arena):
        self.video_name = f'{path_to_beh[:len(path_to_beh) - 4]}'
        self.path_to_data = f'mouse_data\\{self.video_name}_data.xlsx'
        self.path_to_plots = f'mouse_data\\{self.video_name}_plots'
        os.makedirs(self.path_to_plots, exist_ok=True)
        self.df = pd.read_excel(self.path_to_data)
        self.radius_arena = radius_arena
        self.meters_in_px = RADUIS_ARENA_IN_METERS / self.radius_arena


    def plot(self):
        self.plot_trajectory()
        self.plot_speed()
        self.plot_position_heatmap()
        self.plot_velocity_heatmap()
        self.plot_hist_zones()
        self.plot_behs_on_trajectory()
        self.plot_ethogram()

    def plot_trajectory(self):
        start_x, start_y = self.df['X, px'].iloc[0], self.df['Y, px'].iloc[0]
        end_x, end_y = self.df['X, px'].iloc[-1], self.df['Y, px'].iloc[-1]

        plt.figure(figsize=(10, 10))

        plt.plot(self.df['X, px'], self.df['Y, px'], color='darkorange')

        circle = plt.Circle((0, 0), self.radius_arena, color='black', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)

        plt.scatter(start_x, start_y, color='white', edgecolor='black', zorder=5, s=65, label="Start")
        plt.scatter(end_x, end_y, color='black', edgecolor='black', zorder=5, s=65, label="Finish")
        plt.scatter(0, 0, color='red', edgecolor='red', s=75, zorder=5, label="center")

        plt.xlim(-self.radius_arena, self.radius_arena)
        plt.ylim(-self.radius_arena, self.radius_arena)
        plt.gca().set_aspect('equal', adjustable='box')

        xticks_px = plt.gca().get_xticks()
        yticks_px = plt.gca().get_yticks()

        xticks_m = xticks_px * self.meters_in_px 
        yticks_m = yticks_px * self.meters_in_px

        plt.xticks(xticks_px, labels=[f"{tick:.2f}" for tick in xticks_m])  
        plt.yticks(yticks_px, labels=[f"{tick:.2f}" for tick in yticks_m])

        plt.title("Trajectory in Arena")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        filename = os.path.join(self.path_to_plots, f'trajectory_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)


    def plot_speed(self):
        max_ticks = 20  # Максимальное число меток на оси X
        tick_step = max(1, len(self.df) // max_ticks)  # Шаг между метками

        tick_positions = np.arange(0, len(self.df), tick_step)
        tick_labels = self.df["Time, m:s"].iloc[tick_positions]

        window_length = 91
        polyorder = 4

        self.df['Smoothed Speed (Savitzky-Golay)'] = savgol_filter(self.df['Speed, m/s'], window_length, polyorder)

        plt.figure(figsize=(10, 5))
        plt.plot(self.df['Time, m:s'], self.df['Speed, m/s'], color='green', alpha=0.3, label="Original Speed")
        plt.plot(self.df['Time, m:s'], self.df['Smoothed Speed (Savitzky-Golay)'], color='red', label="Smoothed Speed (Savitzky-Golay)")
        plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45, ha='right')  # Повернем для лучшей читаемости
        plt.title("Speed over Time (Smoothed with Savitzky-Golay)")
        plt.xlabel("Time, m:s")
        plt.ylabel("Speed, m/s")
        plt.legend()
        filename = os.path.join(self.path_to_plots, f'speed_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    def plot_position_heatmap(self):
        x = self.df['X, px']
        y = self.df['Y, px']

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=75, range=[[-self.radius_arena, self.radius_arena], [-self.radius_arena, self.radius_arena]])

        heatmap = gaussian_filter(heatmap, sigma=1.5)

        heatmap_log = np.log1p(heatmap)

        plt.figure(figsize=(10, 10))

        extent = [-self.radius_arena, self.radius_arena, -self.radius_arena, self.radius_arena]
        im = plt.imshow(heatmap_log.T, origin='lower', cmap='plasma', interpolation='bilinear', extent=extent)

        circle = plt.Circle((0, 0), self.radius_arena, color='white', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)
        plt.scatter(0, 0, color='white', edgecolor='white', s=75, zorder=5, label="Center")
        plt.xlim(-self.radius_arena, self.radius_arena)
        plt.ylim(-self.radius_arena, self.radius_arena)
        plt.gca().set_aspect('equal', adjustable='box')

        xticks_px = plt.gca().get_xticks()
        yticks_px = plt.gca().get_yticks()

        xticks_m = xticks_px * self.meters_in_px 
        yticks_m = yticks_px * self.meters_in_px

        plt.xticks(xticks_px, labels=[f"{tick:.2f}" for tick in xticks_m])  
        plt.yticks(yticks_px, labels=[f"{tick:.2f}" for tick in yticks_m])

        cbar = plt.colorbar(im, label="Density")
        cbar.set_label("Density")
        plt.title("Position Heatmap (Log Scale)")
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")

        filename = os.path.join(self.path_to_plots, f'position_heatmap_log_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)


    def plot_velocity_heatmap(self):
        x = self.df['X, px']
        y = self.df['Y, px']
        speeds = self.df['Speed, m/s']

        speed_sum, xedges, yedges = np.histogram2d(
            x, y, weights=speeds, bins=75, range=[[-self.radius_arena, self.radius_arena], [-self.radius_arena, self.radius_arena]])
        count, _, _ = np.histogram2d(
            x, y, bins=75, range=[[-self.radius_arena, self.radius_arena], [-self.radius_arena, self.radius_arena]]
        )

        average_speed = np.divide(speed_sum, count, out=np.zeros_like(speed_sum), where=(count > 0))

        average_speed = gaussian_filter(average_speed, sigma=1.5)

        plt.figure(figsize=(10, 10))
        extent = [-self.radius_arena, self.radius_arena, -self.radius_arena, self.radius_arena]
        im = plt.imshow(average_speed.T, origin='lower', cmap='hot', interpolation='bilinear', extent=extent)

        circle = plt.Circle((0, 0), self.radius_arena, color='white', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)

        plt.scatter(0, 0, color='white', edgecolor='white', s=75, zorder=5, label="Center")

        plt.xlim(-self.radius_arena, self.radius_arena)
        plt.ylim(-self.radius_arena, self.radius_arena)
        plt.gca().set_aspect('equal', adjustable='box')
        xticks_px = plt.gca().get_xticks()
        yticks_px = plt.gca().get_yticks()

        xticks_m = xticks_px * self.meters_in_px 
        yticks_m = yticks_px * self.meters_in_px

        plt.xticks(xticks_px, labels=[f"{tick:.2f}" for tick in xticks_m])  
        plt.yticks(yticks_px, labels=[f"{tick:.2f}" for tick in yticks_m])
        cbar = plt.colorbar(im, label="Average Speed (m/s)")
        plt.title("Mouse Velocity Heatmap (Average Speed)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        filename = os.path.join(self.path_to_plots, f'position_average_speed_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    def plot_hist_zones(self):
        def time_to_seconds(time_str):
            minutes, rest = time_str.split(':')
            seconds, milliseconds = rest.split(',')
            return int(minutes) * 60 + int(seconds) + int(milliseconds) / 100

        self.df["Time (s)"] = self.df["Time, m:s"].apply(time_to_seconds)

        self.df["Time Difference (s)"] = self.df["Time (s)"].diff().fillna(0)

        zone_times = (self.df[ZONES_ARENA].multiply(self.df["Time Difference (s)"], axis=0)).sum()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        zone_times.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title("Time spent in the arena zones (s)")
        plt.xlabel("Zones")
        plt.ylabel("Time (s)")
        plt.xticks(rotation=360)

        plt.subplot(1, 2, 2)
        zone_times_percentage = (zone_times / zone_times.sum()) * 100
        zone_times_percentage.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title("Time spent in the arena areas (%)")
        plt.xlabel("Zones")
        plt.ylabel("Percentage of time (%)")
        plt.xticks(rotation=360)
        filename = os.path.join(self.path_to_plots, f'histogram_for_zones_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    def plot_behs_on_trajectory(self):
        filtered_df = self.df[self.df[BEHAVIORS].sum(axis=1) > 0]
        plt.figure(figsize=(6, 6))

        circle = plt.Circle((0, 0), self.radius_arena, color='gray', fill=False, linestyle='--', linewidth=2)
        plt.gca().add_artist(circle)

        sns.scatterplot(x='X, px', y='Y, px', hue=filtered_df[BEHAVIORS].idxmax(axis=1), data=filtered_df, s=25, palette=BEHAVIORS_PALETTE)
        plt.scatter(0, 0, color='black', edgecolor='black', s=75, zorder=5, label="center")

        plt.xlim(-self.radius_arena, self.radius_arena)
        plt.ylim(-self.radius_arena, self.radius_arena)
        plt.gca().set_aspect('equal', adjustable='box')  
        xticks_px = plt.gca().get_xticks()
        yticks_px = plt.gca().get_yticks()

        xticks_m = xticks_px * self.meters_in_px 
        yticks_m = yticks_px * self.meters_in_px

        plt.xticks(xticks_px, labels=[f"{tick:.2f}" for tick in xticks_m])  
        plt.yticks(yticks_px, labels=[f"{tick:.2f}" for tick in yticks_m])


        plt.title("Inter-group Movement Distribution")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel("X, m")
        plt.ylabel("Y, m")
        filename = os.path.join(self.path_to_plots, f'behaviors_on_trajectory_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    def plot_ethogram(self):
        trimmed_df = self.df[BEHAVIORS].iloc[10:-10]

        time_labels = self.df["Time, m:s"].iloc[10:-10:STEP_TIME_FOR_ETHOGRAM]
        time_indices = range(0, len(trimmed_df), STEP_TIME_FOR_ETHOGRAM)

        plt.figure(figsize=(10, 5))
        sns.heatmap(trimmed_df.astype(int).T, cbar=False, cmap="viridis")
        plt.title("Ethogram")
        plt.xlabel("Time")
        plt.xticks(ticks=time_indices, labels=time_labels, rotation=60)
        filename = os.path.join(self.path_to_plots, f'ethogram_{self.video_name}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

