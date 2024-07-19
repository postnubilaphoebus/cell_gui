import math
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QVBoxLayout,
                             QHBoxLayout,
                             QWidget,
                             QSlider,
                             QPushButton,
                             QLineEdit,
                             QLabel,
                             QMessageBox,
                             QTextEdit,
                             QFileDialog,
                             QSplitter)
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
import argparse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import skimage
from skimage.exposure import equalize_hist
from collections import defaultdict

from colormap import glasbey_cmap, glasbey_cmap_rgb, num_colors
from graphicsview import GraphicsView


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


class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        if filename[-3:] == "npy":
            self.image_data = np.load(filename)
        else:
            self.image_data = skimage.io.imread(filename)
        if self.image_data.max() <= 1:
            temp_min = np.min(self.image_data.ravel())
            temp_max = np.max(self.image_data.ravel())
            self.image_data = (self.image_data - temp_min) / (temp_max - temp_min)
            self.image_data = self.image_data * 255.0
        self.z_max = self.image_data.shape[0] - 1
        self.y_max = self.image_data.shape[1] - 1
        self.x_max = self.image_data.shape[2] - 1
        self.z_min = 0
        self.y_min = 0
        self.x_min = 0
        self.min_pixel_intensity = np.min(self.image_data.ravel())
        self.max_pixel_intensity = np.max(self.image_data.ravel())
        self.brush_width = 3
        self.eraser_radius = 2
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.xy_view = GraphicsView(self, "XY")
        self.xz_view = GraphicsView(self, "XZ")
        self.yz_view = GraphicsView(self, "YZ")
        layout.addWidget(self.xy_view)
        layout.addWidget(self.xz_view)
        layout.addWidget(self.yz_view)

        layout_buttons = QHBoxLayout()
        layout_vertical_buttons = QVBoxLayout()

        self.sliderz = QSlider(Qt.Horizontal)
        self.slider_label = QLabel("Z-Planes (1,2)")
        self.sliderz.setRange(0, self.z_max)
        self.sliderz.setSingleStep(1)
        self.sliderz.setTickInterval(1)
        self.sliderz.setTickPosition(QSlider.TicksBelow)
        self.sliderz.setValue(self.z_max // 2)
        layout_buttons.addWidget(self.slider_label)
        layout_buttons.addWidget(self.sliderz)

        self.slidery = QSlider(Qt.Horizontal)
        self.slidery_label = QLabel("Y-Planes (3,4)")
        self.slidery.setRange(0, self.y_max)
        self.slidery.setSingleStep(1)
        self.slidery.setTickInterval(1)
        self.slidery.setTickPosition(QSlider.TicksBelow)
        self.slidery.setValue(self.y_max // 2)
        layout_buttons.addWidget(self.slidery_label)
        layout_buttons.addWidget(self.slidery)

        self.sliderx = QSlider(Qt.Horizontal)
        self.sliderx_label = QLabel("X-Planes (5,6)")
        self.sliderx.setRange(0, self.x_max)
        self.sliderx.setSingleStep(1)
        self.sliderx.setTickInterval(1)
        self.sliderx.setTickPosition(QSlider.TicksBelow)
        self.sliderx.setValue(self.x_max // 2)
        layout_buttons.addWidget(self.sliderx_label)
        layout_buttons.addWidget(self.sliderx)

        self.brush_label = QLabel("Brush Width:")
        layout_buttons.addWidget(self.brush_label)

        self.default_cursor = QCursor(Qt.ArrowCursor)
        self.cursor_pix = QPixmap('paintbrush_icon.png')
        self.cursor_scaled_pix = self.cursor_pix.scaled(QSize(50, 50), Qt.KeepAspectRatio)
        self.brush_cursor = self.default_cursor

        self.brush_text = QLineEdit()
        self.brush_text.setFixedWidth(30)
        self.brush_text.setText(str(self.brush_width))
        self.brush_text.returnPressed.connect(self.updateBrushWidthFromLineEdit)
        layout_buttons.addWidget(self.brush_text)

        self.eraser_radius_label = QLabel("Eraser Radius:")
        layout_buttons.addWidget(self.eraser_radius_label)
        self.eraser_radius_text = QLineEdit()
        self.eraser_radius_text.setFixedWidth(30)
        self.eraser_radius_text.setText(str(self.eraser_radius))
        self.eraser_radius_text.returnPressed.connect(self.updateEraserRadius)
        layout_buttons.addWidget(self.eraser_radius_text)

        self.foreground_button = QPushButton('Foreground (F)', self)
        self.foreground_button.clicked.connect(self.toggleForeground)
        layout_buttons.addWidget(self.foreground_button)

        self.visualization_mode_button = QPushButton('Visualization Mode (J)', self)
        self.visualization_mode_button.clicked.connect(self.visualization_mode)
        layout_buttons.addWidget(self.visualization_mode_button)
        self.visualization_only = False
        self.backup_greyscale = None
        self.backup_color = None

        self.reset_transformations_button = QPushButton('Visualization Mode (J)', self)

        self.background_button = QPushButton('Background (B)', self)
        self.background_button.clicked.connect(self.toggleBackground)
        layout_buttons.addWidget(self.background_button)

        self.eraser_button = QPushButton('Eraser (E)', self)
        self.eraser_button.clicked.connect(self.toggleEraser)
        layout_buttons.addWidget(self.eraser_button)

        self.save_button = QPushButton('Save Masks', self)
        self.save_button.clicked.connect(self.saveMasks)
        layout_vertical_buttons.addWidget(self.save_button)

        self.find_cell_button = QPushButton('Find Cell (C)', self)
        self.find_cell_button.clicked.connect(self.findCell)
        layout_buttons.addWidget(self.find_cell_button)

        self.markers_off_on_button = QPushButton('Markers Off/On (M)', self)
        self.markers_off_on_button.clicked.connect(self.markersOffOn)
        layout_buttons.addWidget(self.markers_off_on_button)

        self.local_contrast_enhancer_button = QPushButton('Local Contrast Enhancer', self)
        self.local_contrast_enhancer_button.clicked.connect(self.localContrastEnhancer)
        self.local_contrast_enhancer_button.setText("Local Contrast Enhancement (L)")
        self.local_contrast_enhancer_enabled = False
        layout_buttons.addWidget(self.local_contrast_enhancer_button)

        self.submit_contrast_button = QPushButton('Submit Contrast', self)
        self.submit_contrast_button.clicked.connect(self.toggle_submit_contrast)
        self.submit_contrast_button.hide()
        layout_buttons.addWidget(self.submit_contrast_button)

        self.view_finder = True
        self.view_finder_button = QPushButton("Hide/Show View Finder (V)", self)
        self.view_finder_button.clicked.connect(self.hide_show_view_finder)
        layout_buttons.addWidget(self.view_finder_button)

        self.open_button = QPushButton('Open Mask (O)', self)
        self.open_button.clicked.connect(self.open_file_dialog)
        layout_vertical_buttons.addWidget(self.open_button)

        self.select_cell_button = QPushButton('Select Cell (S)', self)
        self.select_cell_button.clicked.connect(self.select_cell)
        self.select_cell_enabled = False
        self.new_cell_selected = False
        layout_vertical_buttons.addWidget(self.select_cell_button)

        self.index_control = IndexControlWidget()
        layout_vertical_buttons.addWidget(self.index_control)
        self.index_control.increase_button.clicked.connect(self.update_index_display)
        self.index_control.decrease_button.clicked.connect(self.update_index_display)
        self.current_highest_cell_index = 1

        self.cell_idx_display = TextDisplay()
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        layout_vertical_buttons.addWidget(self.cell_idx_display)
        layout.addLayout(layout_vertical_buttons)
        layout.addLayout(layout_buttons)

        self.sliderz.valueChanged.connect(self.update_xy_view)
        self.sliderz.valueChanged.connect(self.update_yz_view)
        self.sliderz.valueChanged.connect(self.update_xz_view)
        self.slidery.valueChanged.connect(self.update_xz_view)
        self.sliderx.valueChanged.connect(self.update_yz_view)
        self.slidery.valueChanged.connect(self.update_xy_view)
        self.sliderx.valueChanged.connect(self.update_xy_view)
        self.slidery.valueChanged.connect(self.update_yz_view)
        self.sliderx.valueChanged.connect(self.update_xz_view)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
        self.hist_figure = Figure()
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        self.plot_hist()
        self.side_layout.addWidget(self.hist_toolbar)
        self.side_layout.addWidget(self.hist_canvas)
        splitter = QSplitter()
        splitter.addWidget(self.central_widget)
        splitter.addWidget(self.side_widget)
        self.setCentralWidget(splitter)

        self.drawing = False
        self.dragging = False
        self.temp_past_points = []
        self.xy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.xy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.xz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.xz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.yz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.yz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.paint_color = QColor(Qt.red)
        self.paint_color.setAlphaF(0.3)
        self.foreground_enabled = False
        self.background_enabled = False
        self.eraser_enabled = False
        self.last_mouse_pos = None
        self.first_mouse_pos_for_contrast_rect = None
        self.last_mouse_pos_for_contrast_rect = None
        self.yz_view.rotate_view(-90)
        self.xz_view.rotate_view(-90)
        self.foreground_points = []
        self.background_points = []
        self.copied_points = []
        self.markers_enabled = True
        self.update_xz_view()
        self.update_yz_view()
        self.update_xy_view()

    def numpyArrayToPixmap(self, img_np):
        img_np = np.require(img_np, np.uint8, 'C')
        # Check if the image has 3 channels (RGB)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            qim = QImage(img_np.data, img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format_RGB888)
        else:
            qim = QImage(img_np.data, img_np.shape[1],
                         img_np.shape[0], img_np.strides[0],
                         QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def visualization_mode(self):
        if not self.visualization_only:
            self.visualization_mode_button.setStyleSheet("background-color: lightgreen")
            self.repaint()
            img = self.image_data
            rgb_img = np.stack([img] * 3, axis=-1)
            # sort according to color index
            points = sorted(self.foreground_points, key=lambda x: x[-1])
            self.visualization_only = True
            self.backup_greyscale = img
            batches = defaultdict(list)
            for point in points:
                batches[point[-1]].append(point[:3])
            for key in batches:
                points_list = batches[key]
                color_tuple = glasbey_cmap_rgb[key]
                for point in points_list:
                    x, y, z = point
                    rgb_img[z, y, x] = color_tuple
            self.image_data = rgb_img
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        else:
            self.visualization_only = False
            self.visualization_mode_button.setStyleSheet("")
            self.repaint()
            self.backup_color = self.image_data
            self.image_data = self.backup_greyscale
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()

    def hide_show_view_finder(self):
        if self.view_finder:
            self.view_finder_button.setText("View Finder Off (V)")
        else:
            self.view_finder_button.setText("View Finder On (V)")
        self.view_finder = not self.view_finder
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def select_cell(self):
        self.select_cell_enabled = not self.select_cell_enabled
        if self.select_cell_enabled:
            self.current_highest_cell_index = self.index_control.cell_index
            self.select_cell_button.setStyleSheet("background-color: lightgreen")
        else:
            self.select_cell_button.setStyleSheet("")
            self.new_cell_selected = False
            self.index_control.cell_index = self.current_highest_cell_index
            self.update_index_display()
            self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
        self.repaint()

    def update_index_display(self):
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)

    def plot_hist(self):
        ax = self.hist_figure.add_subplot(111)
        ax.hist(self.image_data.ravel(), bins=int(self.max_pixel_intensity),
                range=(self.min_pixel_intensity, self.max_pixel_intensity))
        ax.set_title('Histogram of Pixel Intensities')
        ax.set_ylabel("Bin pixel count")
        ax.set_xlabel("Pixel intensity")
        self.hist_canvas.draw()

    def toggleForeground(self):
        if not self.foreground_enabled:
            self.drawing = True
            self.foreground_enabled = True
            self.background_enabled = False
            self.eraser_enabled = False
            self.foreground_button.setStyleSheet("background-color: lightgreen")
            self.background_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.foreground_enabled = False
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        self.repaint()
        self.central_widget.clearFocus()

    def toggleBackground(self):
        if not self.background_enabled:
            self.drawing = True
            self.foreground_enabled = False
            self.background_enabled = True
            self.eraser_enabled = False
            self.background_button.setStyleSheet("background-color: lightgreen")
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.background_enabled = False
            self.background_button.setStyleSheet("")
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        self.repaint()
        self.central_widget.clearFocus()

    def toggleEraser(self):
        self.foreground_enabled = False
        self.background_enabled = False
        if not self.eraser_enabled:
            self.eraser_enabled = True
            self.drawing = True
            self.eraser_button.setStyleSheet("background-color: lightgreen")
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
            self.central_widget.setCursor(self.default_cursor)
        else:
            self.eraser_enabled = False
            self.drawing = False
            self.eraser_button.setStyleSheet("")
            self.foreground_button.setStyleSheet("")
            self.background_button.setStyleSheet("")
        self.repaint()
        self.central_widget.clearFocus()

    def updateBrushWidthFromLineEdit(self):
        new_width_str = self.brush_text.text()
        try:
            new_width = int(new_width_str)
            if new_width <= 0:
                raise ValueError("Brush width must be a positive integer")
            self.updateBrushWidth(new_width)
        except ValueError:
            self.brush_text.setText(str(self.brush_width))
        finally:
            self.brush_text.clearFocus()

    def updateBrushWidth(self, new_width):
        self.brush_width = new_width

    def updateEraserRadius(self):
        new_radius_str = self.eraser_radius_text.text()
        try:
            new_radius = int(new_radius_str)
            if new_radius <= 0:
                raise ValueError("Eraser radius must be a positive integer")
            self.eraser_radius = new_radius
        except ValueError:
            self.eraser_radius_text.setText(str(self.eraser_radius))
        finally:
            self.eraser_radius_text.clearFocus()

    def removePoints(self, point, view_plane):
        if self.eraser_enabled and self.markers_enabled:
            #if view_plane ....
                # idx = (0,1,)
            #elif
            # idx = (1,2)

            if view_plane == "XY":
                self.foreground_points = [p for p in self.foreground_points \
                                          if (math.dist((point[0], point[1]),
                                                        (p[0], p[1])) > self.eraser_radius or p[2] != point[2]) or \
                                          p[-1] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                          if (math.dist((point[0], point[1]),
                                                        (p[0], p[1])) > self.eraser_radius or p[2] != point[2])]
            elif view_plane == "XZ":
                self.foreground_points = [p for p in self.foreground_points \
                                          if (math.dist((point[0], point[2]),
                                                        (p[0], p[2])) > self.eraser_radius or p[1] != point[1]) or \
                                          p[-1] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                          if (math.dist((point[0], point[2]),
                                                        (p[0], p[2])) > self.eraser_radius or p[1] != point[1])]
            elif view_plane == "YZ":
                self.foreground_points = [p for p in self.foreground_points \
                                          if (math.dist((point[1], point[2]),
                                                        (p[1], p[2])) > self.eraser_radius or p[0] != point[0]) or \
                                          p[-1] != self.index_control.cell_index]
                self.background_points = [p for p in self.background_points \
                                          if (math.dist((point[1], point[2]),
                                                        (p[1], p[2])) > self.eraser_radius or p[0] != point[0])]

    def check_overlaps(self, points):
        points_wo_index = [tuple(point[:-1]) for point in points]
        indices_to_remove = []
        for idx, point in enumerate(points):
            current_points_wo_index = points_wo_index[:idx] + points_wo_index[idx + 1:]
            if tuple(point[:-1]) in current_points_wo_index:
                indices_to_remove.append(idx)
        if indices_to_remove:
            overlapnum = str(len(indices_to_remove))
            mbox = QMessageBox.question(self, 'Warning: {} Points overlap'.format(overlapnum),
                                        "Some points overlap. Do you want to remove them?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                return [points[idx] for idx in range(len(points)) if idx not in indices_to_remove]
            else:
                return points
        else:
            return points

    def saveMasks(self):
        mask = np.zeros_like(self.image_data, dtype=int) - 1
        points = self.foreground_points
        mbox = QMessageBox.question(self, 'Overlaps', "Do you want to check for overlaps? This may take some time",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if mbox == QMessageBox.Yes:
            points = self.check_overlaps(self.foreground_points)
            QMessageBox.about(self, "Overlaps checked", "Overlaps checked")
        for point in self.background_points:
            x, y, z = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = 0
        for point in points:
            x, y, z, idx = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = idx
        filenameNew = self.filename.replace(".", "")
        mask_name = filenameNew + "_mask.npy"
        if os.path.isfile(mask_name):
            mbox = QMessageBox.question(self, 'Warning: Masks file exists', "Do you want to overwrite it?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                np.save(mask_name, mask)
                QMessageBox.about(self, "Masks saved", "Masks saved as %s" % (mask_name))
            else:
                QMessageBox.about(self, "Saving aborted", "Masks not saved")
        else:
            np.save(mask_name, mask)
            QMessageBox.about(self, "Masks saved", "Masks saved as %s" % (mask_name))

    def load_masks(self, filename, load_background=False):
        assert filename.endswith(".npy"), "Mask file must end with .npy"
        mask = np.load(filename)
        assert mask.shape == self.image_data.shape, "Mask shape does not match image shape"
        z_shape, y_shape, x_shape = mask.shape
        self.background_points = []
        self.foreground_points = []
        for z in range(z_shape):
            for y in range(y_shape):
                for x in range(x_shape):
                    if mask[z, y, x] == 0:
                        if load_background:
                            self.background_points.append((x, y, z))
                    elif mask[z, y, x] > 0:
                        self.foreground_points.append((x, y, z, mask[z, y, x], mask[z, y, x] % num_colors))
                        if mask[z, y, x] > self.index_control.cell_index:
                            self.index_control.cell_index = mask[z, y, x]
        # sort according to color index
        self.foreground_points = sorted(self.foreground_points, key=lambda x: x[-1])
        self.current_highest_cell_index = self.index_control.cell_index
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
        self.update_index_display()
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "Numpy Files (*.npy)", options=options)
        if file_name:
            mbox = QMessageBox.question(self, 'Background pixels',
                                        "Do you want to color the background (0s) also? (Not recommended)",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if mbox == QMessageBox.Yes:
                mbox = QMessageBox.question(self, 'Load Background pixels', "Are you sure?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if mbox == QMessageBox.Yes:
                    self.load_masks(file_name, True)
                else:
                    self.load_masks(file_name, False)
            else:
                self.load_masks(file_name, False)
            self.load_masks(file_name)

    def localContrastEnhancer(self):
        if self.local_contrast_enhancer_enabled:
            self.local_contrast_enhancer_enabled = False
            self.first_mouse_pos_for_contrast_rect = None
            self.last_mouse_pos_for_contrast_rect = None
            self.local_contrast_enhancer_button.setStyleSheet("")
            self.submit_contrast_button.hide()
        else:
            self.local_contrast_enhancer_enabled = True
            self.submit_contrast_button.show()
            self.local_contrast_enhancer_button.setText("Local Contrast Enhancer")
            self.local_contrast_enhancer_button.setStyleSheet("background-color: lightgreen")
        self.repaint()
        self.update_xy_view()
        #TODO check?

    def toggle_submit_contrast(self):
        if (
                self.first_mouse_pos_for_contrast_rect and self.last_mouse_pos_for_contrast_rect) and not self.visualization_only:
            x_min = min(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            x_max = max(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            y_min = min(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            y_max = max(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            x_diff = abs(self.last_mouse_pos_for_contrast_rect.x() - self.first_mouse_pos_for_contrast_rect.x())
            y_diff = abs(self.last_mouse_pos_for_contrast_rect.y() - self.first_mouse_pos_for_contrast_rect.y())
            z_height = max(x_diff, y_diff) // 2
            z_min = max(self.sliderz.value() - z_height, 0)
            z_max = min(self.sliderz.value() + z_height, self.z_max)
            subimage = self.image_data[z_min:z_max, y_min:y_max, x_min:x_max]
            eq_subimage = equalize_hist(subimage) * self.max_pixel_intensity
            self.image_data[z_min:z_max, y_min:y_max, x_min:x_max] = eq_subimage
            self.first_mouse_pos_for_contrast_rect = None
            self.last_mouse_pos_for_contrast_rect = None
            self.submit_contrast_button.hide()
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            self.plot_hist()
        else:
            QMessageBox.about(self, "No selection chosen", "%s" % ("Please choose a region for contrast enhancement  \
                                                                   by using the local contrast button for drawing"))

    def findCell(self):
        pts = self.foreground_points
        if pts:
            if not self.visualization_only:
                last_cell_index = pts[-1][3]
                matching_points = [point[:4] for point in pts if point[3] == last_cell_index]
                if len(matching_points) > 2:
                    average_point = np.median(matching_points, axis=0)
                    self.slidery.setValue(round(average_point[1]))
                    self.sliderx.setValue(round(average_point[0]))
                    self.sliderz.setValue(round(average_point[2]))
                else:
                    self.slidery.setValue(pts[-1][1])
                    self.sliderx.setValue(pts[-1][0])
                    self.sliderz.setValue(pts[-1][2])
        else:
            QMessageBox.about(self, "Foreground empty", "%s" % ("Please draw cells using the foreground button"))

    def markersOffOn(self):

        if self.markers_enabled:
            self.markers_enabled = False
            self.markers_off_on_button.setText("Markers Off (M)")
            self.foreground_enabled = False
            self.background_enabled = False
            self.eraser_enabled = False
            if self.visualization_only and self.backup_greyscale is not None:
                self.backup_color = self.image_data
                self.image_data = self.backup_greyscale
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        else:
            self.markers_enabled = True
            self.markers_off_on_button.setText("Markers On (M)")
            if self.visualization_only and self.backup_color is not None:
                self.backup_greyscale = self.image_data
                self.image_data = self.backup_color
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        self.repaint()

    def update_xy_view(self):
        z_index = self.sliderz.value()
        image = self.image_data[z_index]
        pixmap = self.numpyArrayToPixmap(image)
        if self.markers_enabled and not self.visualization_only:

            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[2] == z_index]
                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[0], point[1])

            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[2] == z_index:
                        painter.drawPoint(point[0], point[1])

            if self.view_finder:
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                y_val = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                x_val = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.width())
                square_size = 30
                top_left_x = x_val - square_size // 2
                top_left_y = y_val - square_size // 2
                painter.drawEllipse(top_left_x, top_left_y, square_size, square_size)
                painter.drawLine(x_val, 0, x_val, pixmap.height())
                painter.drawLine(0, y_val, pixmap.width(), y_val)
            painter.end()

        if self.local_contrast_enhancer_enabled and \
                self.first_mouse_pos_for_contrast_rect and \
                self.last_mouse_pos_for_contrast_rect and \
                not self.visualization_only:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(1)
            painter.setPen(pen)
            rect = QRect(self.first_mouse_pos_for_contrast_rect, self.last_mouse_pos_for_contrast_rect)
            painter.drawRect(rect)
            painter.end()
        self.xy_view.setPixmap(pixmap)

    def slider_to_pixmap(self, slider_value, slider_min, slider_max, pixmap_min, pixmap_max):
        return int((slider_value - slider_min) / (slider_max - slider_min) * (pixmap_max - pixmap_min) + pixmap_min)

    def update_xz_view(self):
        y_index = self.slidery.value()
        if self.image_data.ndim == 4:
            image = self.image_data[:, y_index, :, :]
            image = np.transpose(image, (1, 0, 2))
        else:
            image = self.image_data[:, y_index, :].T
        pixmap = self.numpyArrayToPixmap(image)

        if self.markers_enabled and not self.visualization_only:
            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[1] == y_index]

                #for k,v in colormapping.iteritems()
                #k ~= color, qpen
                #v ~= groupofpointstopaint
                #paint.drawPoints(v)
                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[2], point[0])

            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[1] == y_index:
                        painter.drawPoint(point[2], point[0])
            if self.view_finder:
                pixmapx = self.slider_to_pixmap(self.sliderz.value(), 0, self.z_max, 0, pixmap.width())
                pixmapy = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.height())
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

            painter.end()
        self.xz_view.setPixmap(pixmap)

    def update_yz_view(self):
        x_index = self.sliderx.value()
        if self.image_data.ndim == 4:
            image = self.image_data[:, :, x_index, :]
            image = np.transpose(image, (1, 0, 2))
        else:
            image = self.image_data[:, :, x_index].T
        pixmap = self.numpyArrayToPixmap(image)
        if self.markers_enabled and not self.visualization_only:
            painter = QPainter(pixmap)
            if self.foreground_points:
                color_idx = self.foreground_points[0][-1]
                color_count = 0
                pen = QPen(QColor(glasbey_cmap[color_idx]))
                pen.setWidth(1)
                painter.setPen(pen)
                relevant_points = [point for point in self.foreground_points if point[0] == x_index]

                for point in relevant_points:
                    if point[-1] != color_idx:
                        color_idx = point[-1]
                        color_count += 1
                        pen = QPen(QColor(glasbey_cmap[color_idx]))
                        pen.setWidth(1)
                        painter.setPen(pen)
                    painter.drawPoint(point[2], point[1])

            pen = QPen(Qt.blue)
            pen.setWidth(1)
            painter.setPen(pen)

            if self.background_points and not self.visualization_only:
                for point in self.background_points:
                    if point[0] == x_index:
                        painter.drawPoint(point[2], point[1])

            if self.view_finder:
                pixmapy = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                pixmapx = self.slider_to_pixmap(self.sliderz.value(), 0, self.z_max, 0, pixmap.width())
                pen = QPen(QColor(255, 255, 0, 80))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

            painter.end()
        self.yz_view.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    parser = argparse.ArgumentParser(description='Load an image.')
    parser.add_argument('filename', type=str, help='Path to the image to load')
    args = parser.parse_args()
    #TODO - make filename optional, if not there, while filename is none: qfiledialog
    filename = args.filename
    if filename is None:
        parser.error("Please provide a filename.")
        sys.exit(1)
    window = MainWindow(filename)
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle(f'3D TIFF Viewer - {filename}')
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
