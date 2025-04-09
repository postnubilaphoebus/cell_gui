from PyQt5.QtWidgets import (QApplication, 
                             QVBoxLayout, 
                             QHBoxLayout,
                             QWidget, 
                             QSlider, 
                             QPushButton, 
                             QLabel,  
                             QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFormLayout, QSpinBox, QSlider, QDialog
import numpy as np
from tqdm import tqdm
from cmaps import  num_colors
from scipy.ndimage import find_objects

class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the loading screen
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(256, 256)

        # Create a layout and add a label to hold the animation
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setAlignment(Qt.AlignCenter)  # Center the layout

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)  # Center the QLabel within the layout
        layout.addWidget(self.label)

        # Load the spinning wheel GIF with transparency
        self.movie = QMovie("loading.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

        self.setLayout(layout)

        # Center the loading screen in the parent window
        self.center_on_parent()

    def center_on_parent(self):
        if self.parent():
            # Calculate the center position relative to the parent window
            parent_geometry = self.parent().geometry()
            parent_center = parent_geometry.center()
            self.move(parent_center.x() - self.width() // 2,
                      parent_center.y() - self.height() // 2)
        else:
            # No parent, center on the screen
            screen_geometry = QApplication.desktop().screenGeometry()
            screen_center = screen_geometry.center()
            self.move(screen_center.x() - self.width() // 2,
                      screen_center.y() - self.height() // 2)
            

class WatershedDialog(QDialog):
    def __init__(self, image_block, min_shift, threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Watershed Parameters")
        
        self.initial_background_threshold = threshold
        self.current_threshold = threshold
        self.min_shift = min_shift
        self.image_block = image_block
        
        self.layout = QVBoxLayout()
        
        self.slider_label = QLabel(f"Threshold: {self.initial_background_threshold}", self)
        self.layout.addWidget(self.slider_label)
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 255)  
        self.slider.setValue(self.initial_background_threshold)
        self.slider.valueChanged.connect(self.update_preview)
        self.layout.addWidget(self.slider)
        
        self.preview_label = QLabel(self)
        self.layout.addWidget(self.preview_label)
        
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        
        self.setLayout(self.layout)
        
        # Call initial preview update
        self.update_preview()

    def update_preview(self):
        self.current_threshold = self.slider.value()
        self.slider_label.setText(f"Threshold: {self.current_threshold}")
        if self.parent:
            self.parent().watershed_background_threshold = self.current_threshold
            mask = self.image_block > self.current_threshold
            keep_points = np.argwhere(mask) 
            keep_points += self.min_shift
            keep_points = keep_points[:, [2, 1, 0]]
            self.parent().watershed_foreground_points = keep_points
            self.parent().update_xy_view()
            self.parent().update_xz_view()
            self.parent().update_yz_view()

            
class ThresholdDialog(QDialog):
    def __init__(self, image_min, image_max, background_threshold, cell_centre_threshold, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Set Thresholds")
        
        self.image_min = image_min
        self.image_max = image_max
        self.background_threshold = background_threshold
        self.cell_centre_threshold = cell_centre_threshold
        
        # Create layout
        layout = QFormLayout()

        # Create sliders and spin boxes
        self.background_slider = QSlider(Qt.Horizontal, self)
        self.background_slider.setRange(image_min, image_max)
        self.background_slider.setValue(background_threshold)
        
        self.background_spinbox = QSpinBox(self)
        self.background_spinbox.setRange(image_min, image_max)
        self.background_spinbox.setValue(background_threshold)
        
        self.cell_centre_slider = QSlider(Qt.Horizontal, self)
        self.cell_centre_slider.setRange(image_min, image_max)
        self.cell_centre_slider.setValue(cell_centre_threshold)
        
        self.cell_centre_spinbox = QSpinBox(self)
        self.cell_centre_spinbox.setRange(image_min, image_max)
        self.cell_centre_spinbox.setValue(cell_centre_threshold)
        
        # Add widgets to layout
        layout.addRow(QLabel("Background Threshold:"), self.background_slider)
        layout.addRow("", self.background_spinbox)
        layout.addRow(QLabel("Cell Centre Threshold:"), self.cell_centre_slider)
        layout.addRow("", self.cell_centre_spinbox)
        
        # Create and add buttons
        button_layout = QVBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.cancel_button = QPushButton("Cancel", self)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.background_slider.valueChanged.connect(self.background_spinbox.setValue)
        self.background_spinbox.valueChanged.connect(self.background_slider.setValue)
        self.cell_centre_slider.valueChanged.connect(self.cell_centre_spinbox.setValue)
        self.cell_centre_spinbox.valueChanged.connect(self.cell_centre_slider.setValue)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_values(self):
        # Return the adjusted values
        return (self.background_slider.value(), self.cell_centre_slider.value())

class MaskLoader(QThread):
    finished = pyqtSignal() 
    progress = pyqtSignal(int) 

    def __init__(self, parent, filename, load_background=False):
        super().__init__()
        self.parent = parent
        self.filename = filename
        self.load_background = load_background

    def run(self):
        assert self.filename.endswith(".npy"), "Mask file must end with .npy"
        try:
            # mask saved as dict
            mask = np.load(self.filename, allow_pickle='TRUE').item()
            self.parent.background_points = []
            self.parent.foreground_points = []
            self.parent.z_view_dict = {}
            self.parent.y_view_dict = {}
            self.parent.x_view_dict = {}
            print("loading masks...")
            for i in tqdm(range(1, len(mask)+1)):
                global_locs = mask.get(i)
                if global_locs is not None:
                    color_idx = i % num_colors
                    for loc in global_locs:
                        z, y, x = loc
                        self.parent.pure_coordinates.append((x, y, z))
                        self.parent.foreground_points.append((x, y, z, i, color_idx))
                        if z not in self.parent.z_view_dict:
                            self.parent.z_view_dict[z] = []
                        self.parent.z_view_dict[z].append((x, y, i, color_idx))
                        if y not in self.parent.y_view_dict:
                            self.parent.y_view_dict[y] = []
                        self.parent.y_view_dict[y].append((z, x, i, color_idx))
                        if x not in self.parent.x_view_dict:
                            self.parent.x_view_dict[x] = []
                        self.parent.x_view_dict[x].append((z, y, i, color_idx))
                        if self.load_background:
                            self.parent.background_points.append((x, y, z, i, color_idx))
                    if i > self.parent.index_control.cell_index:
                        self.parent.index_control.cell_index = i
                self.progress.emit(i)
        except:
            # mask saved as npy array
            mask = np.load(self.filename)
            if np.issubdtype(mask.dtype, np.floating):
                mask = mask.astype(int)
            assert mask.shape == self.parent.image_data.shape, "Mask shape does not match image shape"
            self.parent.background_points = []
            self.parent.foreground_points = []
            self.parent.z_view_dict = {}
            self.parent.y_view_dict = {}
            self.parent.x_view_dict = {}
            print("loading masks...")
            slices = find_objects(mask)
            for i, slice_tuple in enumerate(tqdm(slices), start=1):
                if slice_tuple is not None:
                    local_locs = np.array(np.where(mask[slice_tuple] == i))
                    global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
                    color_idx = i % num_colors
                    for loc in global_locs:
                        z, y, x = loc
                        self.parent.pure_coordinates.append((x, y, z))
                        self.parent.foreground_points.append((x, y, z, i, color_idx))
                        if z not in self.parent.z_view_dict:
                            self.parent.z_view_dict[z] = []
                        self.parent.z_view_dict[z].append((x, y, i, color_idx))
                        if y not in self.parent.y_view_dict:
                            self.parent.y_view_dict[y] = []
                        self.parent.y_view_dict[y].append((z, x, i, color_idx))
                        if x not in self.parent.x_view_dict:
                            self.parent.x_view_dict[x] = []
                        self.parent.x_view_dict[x].append((z, y, i, color_idx))
                        if self.load_background:
                            self.parent.background_points.append((x, y, z, i, color_idx))
                    if i > self.parent.index_control.cell_index:
                        self.parent.index_control.cell_index = i
                self.progress.emit(i)

        self.parent.foreground_points = sorted(self.parent.foreground_points, key=lambda x: x[-1])
        if self.parent.data_per_tab[self.parent.current_tab_index].get("foreground_points") is not None:
            self.parent.data_per_tab[self.parent.current_tab_index]["foreground_points"] = self.parent.foreground_points
            self.parent.data_per_tab[self.parent.current_tab_index]["pure_coordinates"] = self.parent.pure_coordinates
            self.parent.data_per_tab[self.parent.current_tab_index]["z_view_dict"] = self.parent.z_view_dict
            self.parent.data_per_tab[self.parent.current_tab_index]["y_view_dict"] = self.parent.y_view_dict
            self.parent.data_per_tab[self.parent.current_tab_index]["x_view_dict"] = self.parent.x_view_dict
        self.parent.current_highest_cell_index = self.parent.index_control.cell_index
        self.finished.emit()

class Estimating_Cell_Thresholds(QThread):
    finished = pyqtSignal()  # Signal to indicate the task is finished
    progress = pyqtSignal(int)  # Signal to indicate progress (optional)
    cell_thresholds = pyqtSignal(tuple)

    def __init__(self, image, points):
        super().__init__()
        self.image = image
        self.points = points

    def minimum_of_brightest_spots(self, image, points):
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        cell_indices = points[:, 3]
        unique_cell_indices = np.unique(cell_indices)
        maximum_brightness_per_cell = []
        for cell_idx in unique_cell_indices:
            mask = (cell_indices == cell_idx)
            filtered_x = x_coords[mask]
            filtered_y = y_coords[mask]
            filtered_z = z_coords[mask]
            pixel_values = image[filtered_x, filtered_y, filtered_z]
            max_brightness = np.max(pixel_values)
            maximum_brightness_per_cell.append(max_brightness)

        maximum_brightness_per_cell = np.array(maximum_brightness_per_cell)
        minimum_median_brightness = np.percentile(maximum_brightness_per_cell, 25)
        return minimum_median_brightness

    def run(self):
        print("Estimating thresholds...")
        foreground = [point[:4] for point in self.points]
        foreground = np.array(foreground)
        background_mask = np.ones_like(self.image, dtype=bool)
        background_mask[foreground[:, 0], foreground[:, 1], foreground[:, 2]] = False
        minimum_centre_brightness = self.minimum_of_brightest_spots(self.image, foreground)
        #minimum_centre_brightness *= 0.8
        median_background_value = np.median(background_mask * self.image)
        self.cell_thresholds.emit((minimum_centre_brightness, median_background_value))
        self.finished.emit()

class IndexControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.cell_index = 1
        layout = QHBoxLayout()
        self.decrease_button = QPushButton('<')
        self.decrease_button.clicked.connect(self.decrease_index)
        layout.addWidget(self.decrease_button)
        self.index_label = QLabel(str(self.cell_index))
        layout.addWidget(self.index_label)
        self.increase_button = QPushButton('>')
        self.increase_button.clicked.connect(self.increase_index)
        layout.addWidget(self.increase_button)
        self.setLayout(layout)
        
    def decrease_index(self):
        if self.cell_index == 1:
            return
        self.cell_index -= 1
        self.index_label.setText(str(self.cell_index))
        self.repaint()
        
    def increase_index(self):
        self.cell_index += 1
        self.index_label.setText(str(self.cell_index))
        self.repaint()

    def update_index(self, number, highest_index):
        self.cell_index = number
        self.highest_index = highest_index
        self.index_label.setText(str(self.cell_index) + "/" + str(self.highest_index))
        self.repaint()

class TextDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)  
        self.setMaximumHeight(40)
        self.setMaximumWidth(170)
        self.setAlignment(Qt.AlignCenter)
        
    def update_text(self, number, highest_index):
        self.setText("Selected Cell Index: " + str(number) + " \nHighest Cell Index: " + str(highest_index))
        self.repaint()