import sys
import numpy as np
import os
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
                             QDialog, 
                             QFileDialog,
                             QSplitter,
                             QStackedWidget,
                             QComboBox,
                             QAction,
                             QTabWidget,
                             QMenu)
from PyQt5.QtCore import Qt, QSize, QRect, pyqtSlot, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import skimage
import math
from skimage.exposure import equalize_hist
from collections import defaultdict
from scipy.ndimage import find_objects
from skimage.segmentation import watershed
from tqdm import tqdm
from scipy.ndimage import label, find_objects, distance_transform_edt
from cmaps import glasbey_cmap, glasbey_cmap_rgb
from gui_widgets import *
from graphics_view import GraphicsView
import time
# icon from https://icons8.com/icon/52955/paint

class MainWindow(QMainWindow):
    def __init__(self, filename = None):
        super().__init__()
        self.image_min = 0
        self.image_max = 255
        self.z_max = 10
        self.y_max = 10
        self.x_max = 10
        self.z_min = 0
        self.y_min = 0
        self.x_min = 0
        self.min_pixel_intensity = 0
        self.max_pixel_intensity = 255
        self.image_data = None
        self.filename_list = []
        if filename is not None:
            self.load_image(filename)
            self.filename_list.append(filename)
        self.brush_width = 2
        self.eraser_radius = 2 
        self.num_channels = 0
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.xy_view = None
        self.xz_view = None
        self.yz_view = None
        self.current_zoom_location = None#scene_pos - delta

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.initial_view = 0  
        self.data_per_tab = {}
        self.current_tab_index = 0
        self.tab_indices = []
        self.create_image_view_layout()
        layout.addWidget(self.tab_widget)
        self.current_tab_index = self.tab_widget.currentIndex()

        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        layout_buttons = QVBoxLayout()
        
        self.menu_bar = self.menuBar()

        # Create a menu
        self.file_menu = self.menu_bar.addMenu("File")
        
        # Create actions for the menu
        open_image_action = QAction("Open Image (*.npy, *.tif, *.jpg, *.png)", self)
        open_image_action.triggered.connect(self.open_file)

        open_mask_action = QAction("Open Mask (*.npy)", self)
        open_mask_action.triggered.connect(self.open_mask)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        # Add actions to the menu
        self.file_menu.addAction(open_image_action)
        self.file_menu.addAction(open_mask_action)
        self.file_menu.addAction(save_action)
        self.file_menu.addSeparator()  # Adds a separator line
        self.file_menu.addAction(exit_action)

        self.slider = QSlider(Qt.Horizontal)

        self.slider_label = QLabel("Z-Planes (1,2)")
        self.slider.setRange(0, self.z_max)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setValue(self.z_max//2)
        layout_buttons.addWidget(self.slider_label)
        layout_buttons.addWidget(self.slider)
        
        self.slidery = QSlider(Qt.Horizontal)
        self.slidery_label = QLabel("Y-Planes (3,4)")
        self.slidery.setRange(0, self.y_max)
        self.slidery.setSingleStep(1)
        self.slidery.setTickInterval(1)
        self.slidery.setTickPosition(QSlider.TicksBelow)
        self.slidery.setValue(self.y_max//2)
        layout_buttons.addWidget(self.slidery_label)
        layout_buttons.addWidget(self.slidery)
        
        self.sliderx = QSlider(Qt.Horizontal)
        self.sliderx_label = QLabel("X-Planes (5,6)")
        self.sliderx.setRange(0, self.x_max)
        self.sliderx.setSingleStep(1)
        self.sliderx.setTickInterval(1)
        self.sliderx.setTickPosition(QSlider.TicksBelow)
        self.sliderx.setValue(self.x_max//2)
        layout_buttons.addWidget(self.sliderx_label)
        layout_buttons.addWidget(self.sliderx)

        # self.tab_switch_button = QPushButton("Switch Tab (T)", self)
        # self.tab_switch_button.clicked.connect()
        
        self.brush_label = QLabel("Brush Width:")
        layout_buttons.addWidget(self.brush_label)
        self.default_cursor = QCursor(Qt.ArrowCursor)
        self.cursor_pix = QPixmap('paintbrush_icon.png')
        self.cursor_scaled_pix = self.cursor_pix.scaled(QSize(50, 50), Qt.KeepAspectRatio)
        self.brush_cursor = self.default_cursor

        self.droplet_cursor_pix = QPixmap('droplet_cursor.png')
        self.droplet_cursor_scaled_pix = self.droplet_cursor_pix.scaled(QSize(20, 20), Qt.KeepAspectRatio)
        self.droplet_cursor = QCursor(self.droplet_cursor_scaled_pix)
        
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

        # self.visualization_mode_button = QPushButton('Visualization Mode (J)', self)
        # self.visualization_mode_button.clicked.connect(self.visualization_mode)
        # layout_buttons.addWidget(self.visualization_mode_button)
        self.visualization_only = False
        self.backup_greyscale = None
        self.backup_color = None

        self.background_button = QPushButton('Background (B)', self)
        self.background_button.clicked.connect(self.toggleBackground)
        self.background_button.hide()
        layout_buttons.addWidget(self.background_button)


        self.eraser_button = QPushButton('Eraser (E)', self)
        self.eraser_button.clicked.connect(self.toggleEraser)
        layout_buttons.addWidget(self.eraser_button)

        self.save_button = QPushButton('Save Masks', self)
        self.save_button.clicked.connect(self.saveMasks)
        layout_buttons.addWidget(self.save_button)

        self.find_cell_button = QPushButton('Find Cell (C)', self)
        self.find_cell_button.clicked.connect(self.findCell)
        layout_buttons.addWidget(self.find_cell_button)

        self.label_corrections_button = QPushButton('Label Corrections (X)', self)
        self.label_corrections_button.clicked.connect(self.labelCorrections)
        layout_buttons.addWidget(self.label_corrections_button)

        self.background_threshold = 0.0
        self.cell_centre_threshold = 0.0
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
        layout_buttons.addWidget(self.open_button)

        self.select_cell_button = QPushButton('Select Cell (S)', self)
        self.select_cell_button.clicked.connect(self.select_cell)
        self.select_cell_enabled = False
        self.new_cell_selected = False
        layout_buttons.addWidget(self.select_cell_button)

        self.delete_cell_button = QPushButton('Delete Cell (D)', self)
        self.delete_cell_button.clicked.connect(self.delete_cell)
        layout_buttons.addWidget(self.delete_cell_button)
        self.delete_cell_enabled = False

        self.index_control = IndexControlWidget()
        layout_buttons.addWidget(self.index_control)
        self.index_control.increase_button.clicked.connect(self.update_index_display)
        self.index_control.decrease_button.clicked.connect(self.update_index_display)
        self.current_highest_cell_index = 1

        self.cell_idx_display = TextDisplay()
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        layout_buttons.addWidget(self.cell_idx_display)
        # layout.addLayout(layout_buttons)
        # layout.addLayout(layout_buttons)

        self.progress_label = QLabel('Progress: 0%', self)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        

        self.slider.valueChanged.connect(self.update_xy_view)
        self.slider.valueChanged.connect(self.update_yz_view)
        self.slider.valueChanged.connect(self.update_xz_view)
        self.slidery.valueChanged.connect(self.update_xz_view)
        self.sliderx.valueChanged.connect(self.update_yz_view)
        self.slidery.valueChanged.connect(self.update_xy_view)
        self.sliderx.valueChanged.connect(self.update_xy_view)
        self.slidery.valueChanged.connect(self.update_yz_view)
        self.sliderx.valueChanged.connect(self.update_xz_view)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
        # self.hist_figure = Figure()
        # self.hist_canvas = FigureCanvas(self.hist_figure)
        # self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        # self.side_layout.addWidget(self.hist_toolbar)
        # self.side_layout.addWidget(self.hist_canvas)
        self.side_layout.addLayout(layout_buttons)
        splitter = QSplitter()
        splitter.addWidget(self.central_widget)
        splitter.addWidget(self.side_widget)
        splitter.setSizes([700, 300]) 
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
        self.watershed_neighborhood_selection_enabled = False
        self.watershed_seeding_point = None
        self.watershed_radius = 2
        self.watershed_foreground_points = []
        self.watershed_background_threshold = int(0.5 * 255)
        self.eraser_enabled = False
        self.last_mouse_pos = None
        self.first_mouse_pos_for_contrast_rect = None
        self.last_mouse_pos_for_contrast_rect = None
        self.first_mouse_pos_for_watershed_cube = None
        self.last_mouse_pos_for_watershed_cube = None
        # self.yz_view.rotate_view(-90)
        # self.xz_view.rotate_view(-90)
        self.foreground_points = []
        self.background_points = []
        self.z_view_dict = {}
        self.y_view_dict = {}
        self.x_view_dict = {}
        self.pure_coordinates = []
        self.copied_points = []
        self.markers_enabled = True
        self.update_xz_view()
        self.update_yz_view()
        self.update_xy_view()

    def tab_switch_button_action(self):
        pass

    def create_image_view_layout(self, image_name="Image", image_data=None):
        if self.initial_view == 0:
            # Create a new QWidget for the layout
            image_view_widget = QWidget()

            # Create a layout for the new image stack (same as the existing layout)
            layout = QVBoxLayout(image_view_widget)

            # Create views for XY, XZ, and YZ planes
            xy_view = GraphicsView(self, "XY")
            xz_view = GraphicsView(self, "XZ")
            yz_view = GraphicsView(self, "YZ")

            # Add the views to the layout
            layout.addWidget(xy_view)
            layout.addWidget(xz_view)
            layout.addWidget(yz_view)

            # Add the widget to the tab widget
            self.tab_widget.addTab(image_view_widget, image_name)

            # Optionally store references to the views for further updates
            self.xy_view = xy_view
            self.xz_view = xz_view
            self.yz_view = yz_view
            self.initial_view = 1
        elif self.initial_view == 1:
            self.tab_widget.removeTab(0)
            image_view_widget = QWidget()
            layout = QVBoxLayout(image_view_widget)

            xy_view = GraphicsView(self, "XY")
            xz_view = GraphicsView(self, "XZ")
            yz_view = GraphicsView(self, "YZ")
            yz_view.rotate_view(-90)
            xz_view.rotate_view(-90)


            self.image_min = image_data.min()
            self.image_max = image_data.max()
            self.z_max = image_data.shape[0] - 1
            self.y_max = image_data.shape[1] - 1
            self.x_max = image_data.shape[2] - 1
            self.z_min = 0
            self.y_min = 0
            self.x_min = 0
            self.min_pixel_intensity = self.image_min
            self.max_pixel_intensity = self.image_max
            self.image_data = image_data
            self.brush_width = 2
            self.eraser_radius = 3
            self.xy_view = xy_view
            self.xz_view = xz_view
            self.yz_view = yz_view
            self.filename = image_name
            self.current_highest_cell_index = 0
            self.foreground_points = []
            self.background_points = []
            self.z_view_dict = {}
            self.y_view_dict = {}
            self.x_view_dict = {}
            self.pure_coordinates = []
            self.copied_points = []

            layout.addWidget(xy_view)
            layout.addWidget(xz_view)
            layout.addWidget(yz_view)

            self.tab_widget.insertTab(0, image_view_widget, image_name)
            if 0 not in self.data_per_tab:
                self.data_per_tab[0] = {}
            self.data_per_tab[0] = {
                "xy_view": xy_view,
                "xz_view": xz_view,
                "yz_view": yz_view,
                "image_min": self.image_min,
                "image_max": self.image_max,
                "z_max": self.z_max,
                "y_max": self.y_max,
                "x_max": self.x_max,
                "z_min": self.z_min,
                "y_min": self.y_min,
                "x_min": self.x_min,
                "min_pixel_intensity": self.min_pixel_intensity,
                "max_pixel_intensity": self.max_pixel_intensity,
                "image_data": self.image_data,
                "brush_width": self.brush_width,
                "eraser_radius": self.eraser_radius,
                "filename": self.filename,
                "current_highest_cell_index": self.current_highest_cell_index,
                "foreground_points": self.foreground_points.copy(),
                "background_points": self.background_points.copy(),
                "z_view_dict": self.z_view_dict.copy(),
                "y_view_dict": self.y_view_dict.copy(),
                "x_view_dict": self.x_view_dict.copy(),
                "pure_coordinates": self.pure_coordinates.copy(),
                "copied_points": self.copied_points.copy()
            }
            self.initial_view += 1
        else:
            # Create a new QWidget for the layout
            image_view_widget = QWidget()

            # Create a layout for the new image stack (same as the existing layout)
            layout = QVBoxLayout(image_view_widget)

            # Create views for XY, XZ, and YZ planes
            xy_view = GraphicsView(self, "XY")
            xz_view = GraphicsView(self, "XZ")
            yz_view = GraphicsView(self, "YZ")
            yz_view.rotate_view(-90)
            xz_view.rotate_view(-90)

            # Add the widget to the tab widget
            self.tab_widget.addTab(image_view_widget, image_name)

            
            self.image_min = image_data.min()
            self.image_max = image_data.max()
            self.z_max = image_data.shape[0] - 1
            self.y_max = image_data.shape[1] - 1
            self.x_max = image_data.shape[2] - 1
            self.z_min = 0
            self.y_min = 0
            self.x_min = 0
            self.min_pixel_intensity = self.image_min
            self.max_pixel_intensity = self.image_max
            self.image_data = image_data
            self.brush_width = 2
            self.eraser_radius = 3
            self.xy_view = xy_view
            self.xz_view = xz_view
            self.yz_view = yz_view
            self.filename = image_name
            self.current_highest_cell_index = 0
            self.foreground_points = []
            self.background_points = []
            self.z_view_dict = {}
            self.y_view_dict = {}
            self.x_view_dict = {}
            self.pure_coordinates = []
            self.copied_points = []

            layout.addWidget(xy_view)
            layout.addWidget(xz_view)
            layout.addWidget(yz_view)

            self.tab_widget.addTab(image_view_widget, image_name)
            new_tab_index = self.tab_widget.count() - 1
            if new_tab_index not in self.data_per_tab:
                self.data_per_tab[new_tab_index] = {}
            self.data_per_tab[new_tab_index] = {
                "xy_view": xy_view,
                "xz_view": xz_view,
                "yz_view": yz_view,
                "image_min": self.image_min,
                "image_max": self.image_max,
                "z_max": self.z_max,
                "y_max": self.y_max,
                "x_max": self.x_max,
                "z_min": self.z_min,
                "y_min": self.y_min,
                "x_min": self.x_min,
                "min_pixel_intensity": self.min_pixel_intensity,
                "max_pixel_intensity": self.max_pixel_intensity,
                "image_data": self.image_data,
                "brush_width": self.brush_width,
                "eraser_radius": self.eraser_radius,
                "filename": self.filename,
                "current_highest_cell_index": self.current_highest_cell_index,
                "foreground_points": self.foreground_points.copy(),
                "background_points": self.background_points.copy(),
                "z_view_dict": self.z_view_dict.copy(),
                "y_view_dict": self.y_view_dict.copy(),
                "x_view_dict": self.x_view_dict.copy(),
                "pure_coordinates": self.pure_coordinates.copy(),
                "copied_points": self.copied_points.copy()
            }

    def on_tab_changed(self, index):
        self.current_tab_index = index
        if index > -1 and self.data_per_tab.get(index)is not None:

            self.update_tab_view(index)
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            # if self.current_zoom_location is not None:
            #     self.xy_view.center_on_given_location(self.current_zoom_location)
            #     self.yz_view.center_on_given_location(self.current_zoom_location)
            #     self.xz_view.center_on_given_location(self.current_zoom_location)

    def close_tab(self, index):
        if index > -1:
            self.tab_widget.removeTab(index)

    def update_tab_view(self, index):
        self.image_min = self.data_per_tab[index].get("image_min")
        self.image_max = self.data_per_tab[index].get("image_max")
        self.z_max = self.data_per_tab[index].get("z_max")
        self.y_max = self.data_per_tab[index].get("y_max")
        self.x_max = self.data_per_tab[index].get("x_max")
        self.z_min = self.data_per_tab[index].get("z_min")
        self.y_min = self.data_per_tab[index].get("y_min")
        self.x_min = self.data_per_tab[index].get("x_min")
        self.min_pixel_intensity = self.data_per_tab[index].get("min_pixel_intensity")
        self.max_pixel_intensity = self.data_per_tab[index].get("max_pixel_intensity")
        self.image_data = self.data_per_tab[index].get("image_data")
        self.brush_width = self.data_per_tab[index].get("brush_width")
        self.eraser_radius = self.data_per_tab[index].get("eraser_radius")
        self.num_channels = self.data_per_tab[index].get("num_channels")
        self.xy_view = self.data_per_tab[index].get("xy_view")
        self.xz_view = self.data_per_tab[index].get("xz_view")
        self.yz_view = self.data_per_tab[index].get("yz_view")
        self.filename = self.data_per_tab[index].get("filename")
        self.current_highest_cell_index = self.data_per_tab[index].get("current_highest_cell_index")
        self.foreground_points = self.data_per_tab[index].get("foreground_points")
        self.background_points = self.data_per_tab[index].get("background_points")
        self.z_view_dict = self.data_per_tab[index].get("z_view_dict")
        self.y_view_dict = self.data_per_tab[index].get("y_view_dict")
        self.x_view_dict = self.data_per_tab[index].get("x_view_dict")
        self.pure_coordinates = self.data_per_tab[index].get("pure_coordinates")
        self.copied_points = []

        
    def numpyArrayToPixmap(self, img_np):
        img_np = np.require(img_np, np.uint8, 'C')
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            qim = QImage(img_np.data, img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format_RGB888)
        else:
            qim = QImage(img_np.data, img_np.shape[1], 
                         img_np.shape[0], img_np.strides[0], 
                         QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def load_image(self, filename):
        # Load image data
        self.filename = filename
        if filename.endswith("npy"):
            image_data = np.load(filename)
            assert image_data.ndim == 3 or image_data.ndim == 4, "Image must be 3D or 4D"
        else:
            image_data = skimage.io.imread(filename)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        image_data = (image_data * 255.0).astype(np.uint8)
        self.filename_list.append(filename)

        # Update sliders with new image dimensions
        self.z_max = image_data.shape[0] - 1
        self.y_max = image_data.shape[1] - 1
        self.x_max = image_data.shape[2] - 1
        self.slidery.setRange(0, self.y_max)
        self.sliderx.setRange(0, self.x_max)
        self.slider.setRange(0, self.z_max)
        self.slidery.setValue(image_data.shape[1] // 2)
        self.sliderx.setValue(image_data.shape[2] // 2)
        self.slider.setValue(image_data.shape[0] // 2)

        # Create a new tab with the image layout
        image_name = filename.split('/')[-1]  # Use the filename as the tab name
        self.create_image_view_layout(image_name, image_data)

        # # Handle 3D or 4D images (make 3D images RGB by stacking)
        # if image_data.ndim == 3:
        #     image_data = np.stack([image_data] * 3, axis=-1) 

        # Update image parameters and views
        self.image_min = 0
        self.image_max = 255
        self.min_pixel_intensity = np.min(self.image_data.ravel())
        self.max_pixel_intensity = np.max(self.image_data.ravel())
        self.update_xy_view()
        self.update_yz_view()
        self.update_xz_view()

    
    # def load_image(self, filename):
    #     self.filename = filename
    #     if filename[-3:] == "npy":
    #         self.image_data = np.load(filename)
    #         assert self.image_data.ndim == 3 or self.image_data.ndim == 4, "Image must be 3D or 4D"
    #     else:
    #         self.image_data = skimage.io.imread(filename)
    #     self.image_data = (self.image_data - self.image_data.min()) / (self.image_data.max() - self.image_data.min())
    #     self.image_data = (self.image_data * 255.0).astype(np.uint8)
    #     self.z_max = self.image_data.shape[0] - 1
    #     self.y_max = self.image_data.shape[1] - 1
    #     self.x_max = self.image_data.shape[2] - 1
    #     self.slidery.setRange(0, self.y_max)
    #     self.sliderx.setRange(0, self.x_max)
    #     self.slider.setRange(0, self.z_max)
    #     self.slidery.setValue(self.image_data.shape[1] // 2)
    #     self.sliderx.setValue(self.image_data.shape[2] // 2)
    #     self.slider.setValue(self.image_data.shape[0] // 2)
    #     if self.image_data.ndim == 3:
    #         self.image_data = np.stack([self.image_data] * 3, axis=-1) 
    #     self.image_min = 0
    #     self.image_max = 255
    #     self.z_min = 0
    #     self.y_min = 0
    #     self.x_min = 0
    #     self.min_pixel_intensity = np.min(self.image_data.ravel())
    #     self.max_pixel_intensity = np.max(self.image_data.ravel())
    #     self.update_xy_view()
    #     self.update_yz_view()
    #     self.update_xz_view()
    
    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.tif *.jpg *.png);;NumPy Files (*.npy);;All Files (*)",
            options=options
        )
        if file_name:
            self.load_image(file_name)

    def open_mask(self):
        if self.image_data is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "Numpy Files (*.npy)", options=options)
            if file_name:
                self.load_masks(file_name, False)
        else:
            QMessageBox.information(self, "No Image Found", "Please load an image first!")

    def save_file(self):
        mask = np.zeros_like(self.image_data, dtype=int) 
        mbox = QMessageBox.question(self, 'Overlaps', "Do you want to check for overlaps? This may take some time", 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if mbox == QMessageBox.Yes:
            points = self.check_overlaps(self.foreground_points)
            QMessageBox.about(self, "Overlaps checked", "Overlaps checked")
        points = self.foreground_points
        for point in self.background_points:
            x, y, z = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = 0
        for point in points:
            x, y, z, idx, color_idx = point
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

    
    def on_selection_change(self, index):
        print(f"Selected index: {index}, Item: {self.combo_box.currentText()}")
    
    def synchronize_wheeling(self, missing_view_planes, wheel_event):
        graphics_views = [self.xy_view, self.xz_view, self.yz_view]
        for view in graphics_views:
            if view.view_plane in missing_view_planes:
                view.wheelEvent(wheel_event, False)
    
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

    def delete_cell(self):
        self.delete_cell_enabled = not self.delete_cell_enabled
        if self.delete_cell_enabled:
            self.delete_cell_button.setStyleSheet("background-color: lightgreen")
        else:
            self.delete_cell_button.setStyleSheet("")
        self.repaint()

    def apply_watershed(self):
        self.watershed_neighborhood_selection_enabled = not self.watershed_neighborhood_selection_enabled
        radius = self.watershed_radius
        seeding_point = self.watershed_seeding_point
        threshold = self.watershed_background_threshold
        self.watershed_seeding_point = None
        self.watershed_radius = 2
        image_block = self.image_data[seeding_point[2] - radius:seeding_point[2] + radius + 1,
                                      seeding_point[1] - radius:seeding_point[1] + radius + 1,
                                      seeding_point[0] - radius:seeding_point[0] + radius + 1]
        skimage.io.imsave("image_block.tif", image_block)
        min_shift = np.array([seeding_point[2] - radius, seeding_point[1] - radius, seeding_point[0] - radius])
        dialog = WatershedDialog(image_block, min_shift, threshold, self)
        if dialog.exec_() == QDialog.Accepted:
            pass

    def labelCorrections(self):
        mbox = QMessageBox(self)
        if self.foreground_points:
            mbox.setWindowTitle("Correction Type")
            mbox.setText("Would you like to...?")

            button1 = mbox.addButton("Add cells missed by the labeler", QMessageBox.ActionRole)
            button2 = mbox.addButton("Correct already labeled cells", QMessageBox.ActionRole)
            button3 = mbox.addButton("Individual Cell Correction Through Watershed", QMessageBox.ActionRole)
            mbox.exec_()

            if mbox.clickedButton() == button1:
                mbox = QMessageBox(self)
                mbox.setWindowTitle("Labeling estimation")
                mbox.setText("Please enter your labeling setings in the following dialog. \n"
                            "You have to choose among minimum centre brightness and background threshold. "
                            "We have estimated these values based on the current image.\n")
                estimating_cell_thread = Estimating_Cell_Thresholds(self.image_data, self.foreground_points)
                estimating_cell_thread.progress.connect(self.handle_progress)
                estimating_cell_thread.cell_thresholds.connect(self.handle_thresholds)
                estimating_cell_thread.finished.connect(self.handle_finished)
                estimating_cell_thread.start()
                mbox.exec_()

                dialog = ThresholdDialog(self.image_min, self.image_max, self.background_threshold, self.cell_centre_threshold, self)
                if dialog.exec_() == QDialog.Accepted:
                    background_threshold, cell_centre_threshold = dialog.get_values()
                    # Update the class attributes or use the values as needed
                    self.background_threshold = background_threshold
                    self.cell_centre_threshold = cell_centre_threshold
                    foreground_points = [point[:3] for point in self.foreground_points]
                    foreground_points = np.array(foreground_points)
                    foreground_mask = np.zeros_like(self.image_data)
                    background_mask = np.ones_like(self.image_data)
                    foreground_mask[foreground_points[:, 0], foreground_points[:, 1], foreground_points[:, 2]] = 1
                    background_mask = (background_mask - foreground_mask).astype(int)
                    potential_new_cells = self.image_data > cell_centre_threshold
                    potential_new_cells[7:57, 7:57, 7:57] = False
                    labeled_array, num_features = label(potential_new_cells)
                    slices = find_objects(labeled_array)
                    seeding_points = []
                    for i, slice_tuple in enumerate(tqdm(slices)):
                        if slice_tuple is not None:
                            mom_locs = np.array(np.where(labeled_array[slice_tuple] == (i + 1)))
                            mom_locs_global = np.stack(mom_locs).T + np.array([s.start for s in slice_tuple])
                            brightest_within_loc = np.argmax(labeled_array[mom_locs_global[:, 0], mom_locs_global[:, 1], mom_locs_global[:, 2]])
                            brightest_coordinates = mom_locs_global[brightest_within_loc]
                            seeding_points.append(brightest_coordinates)
                            #print("brightest_coordinates", brightest_coordinates)
                    print("seeding_points", seeding_points)
                    seeding_points = np.array(seeding_points)
                    #print("seeding_points", seeding_points)
                    inverted_image = np.ones_like(self.image_data) * 255.0 - self.image_data
                    local_peaks_in_array = np.zeros_like(background_mask).astype(int)
                    for idx, point in enumerate(seeding_points, start=1):
                        local_peaks_in_array[point[0], point[1], point[2]] = idx
                    background_image = (self.image_data > background_threshold).astype(int)
                    distance = distance_transform_edt(background_image)
                    wts = watershed(-distance, local_peaks_in_array, mask=inverted_image)
                    print("wts.max()", wts.max())
                    wts *= background_image
                    # watershed_locs = find_watershed_locations(wts, 50)
                    # filtered_watershed_locs = filter_watershed_locations(watershed_locs, 50)
                    # #print("filtered_watershed_locs", filtered_watershed_locs)
                    # final_watershedded = refine_watershed(filtered_watershed_locs, wts.shape, 50)
                    # #final_watershedded = background_mask * background_image * final_watershedded
                    # locs = find_watershed_locations(final_watershedded, 50)
                    # current_index = 1
                    # added_new_cells = 0
                    # self.foreground_points = []
                    # for points in locs:
                    #     for point in points:
                    #         self.foreground_points.append((point[0], point[1], point[2], current_index, current_index%num_colors))
                    #     current_index += 1
                    #     added_new_cells += 1
                    #print("added_new_cells", added_new_cells)
                    self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
                    self.index_control.update_index(self.index_control.cell_index, self.current_highest_cell_index)
                    self.update_index_display()
                    self.update_xy_view()
                    self.update_xz_view()
                    self.update_yz_view()
            elif mbox.clickedButton() == button2:
                print("Option 2 selected")
            elif mbox.clickedButton() == button3:
                print("Option 3 selected")
                self.watershed_neighborhood_selection_enabled = True
                self.central_widget.setCursor(self.droplet_cursor)
                QMessageBox.about(self, "Watershed Active", "Please select the center of a cell to be filled." 
                    + "\n You can control the neighborhood size with + and -, cancel with right click." 
                    + "\n Confirm your selection with enter.")
        else:
            QMessageBox.about(self, "No foreground cells", "Please label some cells first.")

    @pyqtSlot(int)
    def handle_progress(self, progress):
        self.progress_label.setVisible(True)
        self.progress_label.setText(f"Progress: {progress:.1f}%")

    @pyqtSlot(tuple)
    def handle_thresholds(self, result):
        # Handle the result from the worker thread
        centre_threshold, background_threshold = result
        self.cell_centre_threshold = round(centre_threshold)
        self.background_threshold = round(background_threshold)
        print("Centre Threshold: ", self.cell_centre_threshold )
        print("Background Threshold: ", self.background_threshold)

    @pyqtSlot()
    def handle_finished(self):
        self.progress_label.setVisible(False)

    def update_index_display(self):
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
    
    def plot_hist(self):
        ax = self.hist_figure.add_subplot(111)
        ax.hist(self.image_data.ravel(), bins=int(self.max_pixel_intensity), range=(self.min_pixel_intensity, self.max_pixel_intensity))
        ax.set_title('Histogram of Pixel Intensities')
        ax.set_ylabel("Bin pixel count")
        ax.set_xlabel("Pixel intensity")
        # Define a function for metric formatting
        def metric_format(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.0f}M'  
            elif x >= 1e3:
                return f'{x*1e-3:.0f}K'  
            else:
                return f'{x:.0f}'  

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(metric_format))
        self.hist_canvas.draw()
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

    def remove_indices(self, indices):
        self.foreground_points = [p for i, p in enumerate(self.foreground_points) if i not in indices]
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

    def removePoints(self, point, view_plane):
        if self.eraser_enabled and self.markers_enabled:
            # Determine the coordinate indices based on the view plane
            
            if view_plane == "XY":
                first_index, second_index, third_index = 0, 1, 2
                z_plane_points = self.z_view_dict.get(point[2])
                if not z_plane_points:
                    return
                points_to_remove = [(p[0], p[1], point[2], p[2], p[3]) for p in z_plane_points if (p[3] == point[4] and math.dist((point[0], point[1]), (p[0], p[1])) <= self.eraser_radius)]
            elif view_plane == "XZ":
                first_index, second_index, third_index = 0, 2, 1
                y_plane_points = self.y_view_dict.get(point[1])
                if not y_plane_points:
                    return
                points_to_remove = [(p[1], point[1], p[0], p[2], p[3]) for p in y_plane_points if (p[3] == point[4] and math.dist((point[0], point[2]), (p[1], p[0])) <= self.eraser_radius)]
            elif view_plane == "YZ":
                first_index, second_index, third_index = 1, 2, 0
                x_plane_points = self.x_view_dict.get(point[0])
                if not x_plane_points:
                    return
                points_to_remove = [(point[0], p[1], p[0], p[2], p[3]) for p in x_plane_points if (p[3] == point[4] and math.dist((point[1], point[2]), (p[1], p[0])) <= self.eraser_radius)]
            else:
                return  # Invalid view_plane, nothing to remove
            
            # points_to_remove = [
            #     p for p in self.foreground_points
            #     if (math.dist((point[first_index], point[second_index]), 
            #                     (p[first_index], p[second_index])) <= self.eraser_radius and 
            #         p[third_index] == point[third_index] and 
            #         p[-2] == self.index_control.cell_index)
            # ]
            pure_points_to_remove = [p[:3] for p in points_to_remove]

            self.foreground_points = [
                p for p in self.foreground_points if p not in points_to_remove
            ]
            self.background_points = [
                p for p in self.background_points if p not in points_to_remove
            ]
            
            for p in points_to_remove:
                # Remove from z_view_dict
                if p[2] in self.z_view_dict:
                    self.z_view_dict[p[2]] = [
                        z_point for z_point in self.z_view_dict[p[2]] 
                        if z_point[0] != p[0] or z_point[1] != p[1] or z_point[2] != p[3] or z_point[3] != p[4]
                    ]
                    if not self.z_view_dict[p[2]]:
                        del self.z_view_dict[p[2]]

                # Remove from y_view_dict
                if p[1] in self.y_view_dict:
                    self.y_view_dict[p[1]] = [
                        y_point for y_point in self.y_view_dict[p[1]] 
                        if y_point[1] != p[0] or y_point[0] != p[2] or y_point[2] != p[3] or y_point[3] != p[4]
                    ]
                    if not self.y_view_dict[p[1]]:
                        del self.y_view_dict[p[1]]

                # Remove from x_view_dict
                if p[0] in self.x_view_dict:
                    self.x_view_dict[p[0]] = [
                        x_point for x_point in self.x_view_dict[p[0]] 
                        if x_point[1] != p[1] or x_point[0] != p[2] or x_point[2] != p[3] or x_point[3] != p[4]
                    ]
                    if not self.x_view_dict[p[0]]:
                        del self.x_view_dict[p[0]]

                # remove from pure coordinates
                self.pure_coordinates = [p for p in self.pure_coordinates if p not in pure_points_to_remove]

    def check_overlaps(self, points, suspicious_size = 11):
        new_labels = {}
        new_label_idx = 1
        cell_dict = {}
        for point in points:
            if point[-2] not in cell_dict:
                cell_dict[point[-2]] = []
            cell_dict[point[-2]].append(point)
        for cell in cell_dict.values():
            point_locs = cell[:3]
            minimum_locs = np.min(point_locs, axis = 0)
            maximum_locs = np.max(point_locs, axis = 0)
            diff = maximum_locs - minimum_locs

            if (diff > suspicious_size).sum() > 0:
                cube_shape = diff + [3, 3, 3]
                cube = np.zeros(cube_shape)
                shifted_locs = point_locs + 1  # Shift locs to account for padding
                cube[shifted_locs[:, 0], shifted_locs[:, 1], shifted_locs[:, 2]] = 1
                labeled_cube, num_feats = label(cube)
                if num_feats > 1:
                    objects = find_objects(labeled_cube)  
                    for ii in range(num_feats):
                        if objects[ii] is not None:  
                            obj_slice = objects[ii]
                            cube_locs = np.argwhere(labeled_cube[obj_slice] == ii + 1)
                            cube_locs = cube_locs + np.array([s.start for s in obj_slice]) + minimum_locs - 1
                            if new_label_idx not in new_labels:
                                new_labels[new_label_idx] = []
                            new_labels[new_label_idx].append(cube_locs)
                            new_label_idx += 1
                else:
                    if new_label_idx not in new_labels:
                        new_labels[new_label_idx] = []
                    new_labels[new_label_idx].append(point_locs)
                    new_label_idx += 1
            else:
                if new_label_idx not in new_labels:
                    new_labels[new_label_idx] = []
                new_labels[new_label_idx].append(point_locs)
                new_label_idx += 1



        points_wo_index = [tuple(point[:-1]) for point in points]
        indices_to_remove = []
        for idx, point in enumerate(points):
            current_points_wo_index = points_wo_index[:idx] + points_wo_index[idx+1:]
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
        mask = np.zeros_like(self.image_data, dtype=int) 
        mbox = QMessageBox.question(self, 'Overlaps', "Do you want to check for overlaps and disconnected labels? (recommended)", 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if mbox == QMessageBox.Yes:
            points = self.check_overlaps(self.foreground_points)
            QMessageBox.about(self, "Overlaps checked", "Overlaps checked")
        points = self.foreground_points
        for point in self.background_points:
            x, y, z = point
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = 0
        for point in points:
            x, y, z, idx, color_idx = point
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
        
    def load_masks(self, filename, load_background = False):
        # Show the loading screen
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()

        # Create and start the worker thread
        self.mask_loader = MaskLoader(self, filename, load_background)
        self.mask_loader.finished.connect(self.on_masks_loaded)
        self.mask_loader.start()

    def add_points(self, point, category = "foreground"):
        if isinstance(point, list):
            if category == "foreground":
                for p in point:
                    self.pure_coordinates.append(p[:3])                        
                    self.foreground_points.append(p)
                    if p[2] not in self.z_view_dict:
                        self.z_view_dict[p[2]] = []
                    if p[1] not in self.y_view_dict:
                        self.y_view_dict[p[1]] = []
                    if p[0] not in self.x_view_dict:
                        self.x_view_dict[p[0]] = []
                    self.z_view_dict[p[2]].append((p[0], p[1], p[3], p[4]))
                    self.y_view_dict[p[1]].append((p[2], p[0], p[3], p[4]))
                    self.x_view_dict[p[0]].append((p[2], p[1], p[3], p[4]))
            else:
                for p in point:
                    self.pure_coordinates.append(p)
                    self.background_points.append(p)
        else:
            if category == "foreground":
                self.pure_coordinates.append(point[:3])                        
                self.foreground_points.append(point)
                if point[2] not in self.z_view_dict:
                    self.z_view_dict[point[2]] = []
                if point[1] not in self.y_view_dict:
                    self.y_view_dict[point[1]] = []
                if point[0] not in self.x_view_dict:
                    self.x_view_dict[point[0]] = []
                self.z_view_dict[point[2]].append((point[0], point[1], point[3], point[4]))
                self.y_view_dict[point[1]].append((point[2], point[0], point[3], point[4]))
                self.x_view_dict[point[0]].append((point[2], point[1], point[3], point[4]))
            else:
                self.pure_coordinates.append(point)
                self.background_points.append(point)

    def update_masks(self, action_type, points):
        if action_type == "add":
            self.addPoints(points, "XY")
            self.addPoints(points, "XZ")
            self.addPoints(points, "YZ")
        elif action_type == "remove":
            self.removePoints(points, "XY")
            self.removePoints(points, "XZ")
            self.removePoints(points, "YZ")
        elif action_type == "merge":
            self.mergePoints(points, "XY")
            self.mergePoints(points, "XZ")
            self.mergePoints(points, "YZ")
        elif action_type == "split":
            self.splitPoints(points, "XY")
            self.splitPoints(points, "XZ")
            self.splitPoints(points, "YZ")

    def on_masks_loaded(self):
        self.loading_screen.hide()
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
            self.load_masks(file_name, False)

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

    def toggle_submit_contrast(self):
        if (self.first_mouse_pos_for_contrast_rect and self.last_mouse_pos_for_contrast_rect) and not self.visualization_only:
            x_min = min(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            x_max = max(self.first_mouse_pos_for_contrast_rect.x(), self.last_mouse_pos_for_contrast_rect.x())
            y_min = min(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            y_max = max(self.first_mouse_pos_for_contrast_rect.y(), self.last_mouse_pos_for_contrast_rect.y())
            x_diff = abs(self.last_mouse_pos_for_contrast_rect.x() - self.first_mouse_pos_for_contrast_rect.x())
            y_diff = abs(self.last_mouse_pos_for_contrast_rect.y() - self.first_mouse_pos_for_contrast_rect.y())
            z_height = max(x_diff, y_diff) // 2
            z_min = max(self.slider.value() - z_height, 0)
            z_max = min(self.slider.value() + z_height, self.z_max)
            subimage = self.image_data[z_min:z_max, y_min:y_max, x_min:x_max]
            eq_subimage = equalize_hist(subimage) * self.max_pixel_intensity
            self.image_data[z_min:z_max, y_min:y_max, x_min:x_max] = eq_subimage
            self.first_mouse_pos_for_contrast_rect = None
            self.last_mouse_pos_for_contrast_rect = None
            self.submit_contrast_button.hide()
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            #self.plot_hist()
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
                    self.slider.setValue(round(average_point[2]))
                else:
                    self.slidery.setValue(pts[-1][1])
                    self.sliderx.setValue(pts[-1][0])
                    self.slider.setValue(pts[-1][2])
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
        if self.image_data is not None:
            z_index = self.slider.value()
            image = self.get_flat_image_view("XY", z_index)
            pixmap = self.numpyArrayToPixmap(image)
            if self.markers_enabled and not self.visualization_only:

                painter = QPainter(pixmap)
                if self.foreground_points:
                    color_idx = 0
                    color_count = 0
                    pen = QPen(QColor(glasbey_cmap[color_idx]))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    #relevant_points = [point for point in self.foreground_points if point[2] == z_index]
                    relevant_points = self.z_view_dict.get(z_index)
                    if relevant_points:
                        for point in relevant_points:
                            if point[-1] != color_idx:
                                color_idx = point[-1]
                                color_count += 1
                                pen = QPen(QColor(glasbey_cmap[color_idx]))
                                pen.setWidth(1)
                                painter.setPen(pen)
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

            if self.watershed_neighborhood_selection_enabled and self.watershed_seeding_point:
                xval = self.watershed_seeding_point[0]
                yval = self.watershed_seeding_point[1]
                radius = self.watershed_radius
                if xval - radius >= 0 and yval - radius >= 0 and xval + radius < self.x_max and yval + radius < self.y_max:
                    painter = QPainter(pixmap)
                    pen = QPen(Qt.red)
                    pen.setWidth(1)
                    painter.setPen(pen)
                    front_top_left = QPointF(xval - radius, yval - radius)
                    front_bottom_right = QPointF(xval + radius, yval + radius)
                    painter.drawRect(QRectF(front_top_left, front_bottom_right))
                    painter.end()

            if len(self.watershed_foreground_points) > 0:
                painter = QPainter(pixmap)
                pen = QPen(Qt.red)
                pen.setWidth(1)
                painter.setPen(pen)
                for point in self.watershed_foreground_points:
                    if point[2] == z_index:
                        painter.drawPoint(point[0], point[1])
                painter.end()
            self.xy_view.setPixmap(pixmap)

    def slider_to_pixmap(self, slider_value, slider_min, slider_max, pixmap_min, pixmap_max):
        return int((slider_value - slider_min) / (slider_max - slider_min) * (pixmap_max - pixmap_min) + pixmap_min)
    
    def get_flat_image_view(self, view_plane, flat_index):
        if view_plane == "XY":
            image = self.image_data[flat_index]
        elif view_plane == "XZ":
            if self.image_data.ndim == 4:
                image = self.image_data[:, flat_index, :, :]
                image = np.transpose(image, (1, 0, 2))
            else:
                image = self.image_data[:, flat_index, :].T
        elif view_plane == "YZ":
            if self.image_data.ndim == 4:
                image = self.image_data[:, :, flat_index, :]
                image = np.transpose(image, (1, 0, 2))
            else:
                image = self.image_data[:, :, flat_index].T
        else:
            return
        return image

    def update_xz_view(self):   
        if self.image_data is not None:
            y_index = self.slidery.value()
            image = self.get_flat_image_view("XZ", y_index)
            pixmap = self.numpyArrayToPixmap(image)

            if self.markers_enabled and not self.visualization_only:
                painter = QPainter(pixmap)
                if self.foreground_points:
                    color_idx = 0
                    color_count = 0
                    pen = QPen(QColor(glasbey_cmap[color_idx]))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    #relevant_points = [point for point in self.foreground_points if point[1] == y_index]
                    relevant_points = self.y_view_dict.get(y_index)
                    if relevant_points:
                        for point in relevant_points:
                            if point[-1] != color_idx:
                                color_idx = point[-1]
                                color_count += 1
                                pen = QPen(QColor(glasbey_cmap[color_idx]))
                                pen.setWidth(1)
                                painter.setPen(pen)
                            #painter.drawPoint(point[2], point[0])
                            painter.drawPoint(point[0], point[1])


                pen = QPen(Qt.blue)
                pen.setWidth(1)
                painter.setPen(pen)

                # if self.background_points and not self.visualization_only:
                #     for point in self.background_points:
                #         if point[1] == y_index:
                #             painter.drawPoint(point[2], point[0])
                if self.view_finder:
                    pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                    pixmapy = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.height())
                    pen = QPen(QColor(255, 255, 0, 80))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                    painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

                painter.end()

            if len(self.watershed_foreground_points) > 0:
                painter = QPainter(pixmap)
                pen = QPen(Qt.red)
                pen.setWidth(1)
                painter.setPen(pen)
                for point in self.watershed_foreground_points:
                    if point[1] == y_index:
                        painter.drawPoint(point[2], point[0])

                painter.end()
            self.xz_view.setPixmap(pixmap)

    def update_yz_view(self):
        if self.image_data is not None:
            x_index = self.sliderx.value()
            image = self.get_flat_image_view("YZ", x_index)
            pixmap = self.numpyArrayToPixmap(image)
            if self.markers_enabled and not self.visualization_only:
                painter = QPainter(pixmap)
                if self.foreground_points:
                    color_idx = 0
                    color_count = 0
                    pen = QPen(QColor(glasbey_cmap[color_idx]))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    #relevant_points = [point for point in self.foreground_points if point[0] == x_index]
                    relevant_points = self.x_view_dict.get(x_index)
                    if relevant_points:
                        for point in relevant_points:
                            if point[-1] != color_idx:
                                color_idx = point[-1]
                                color_count += 1
                                pen = QPen(QColor(glasbey_cmap[color_idx]))
                                pen.setWidth(1)
                                painter.setPen(pen)
                            #painter.drawPoint(point[2], point[1])
                            painter.drawPoint(point[0], point[1])

                pen = QPen(Qt.blue)
                pen.setWidth(1)
                painter.setPen(pen)

                # if self.background_points and not self.visualization_only:
                #     for point in self.background_points:
                #         if point[0] == x_index:
                #             painter.drawPoint(point[2], point[1])

                if self.view_finder:    
                    pixmapy = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                    pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                    pen = QPen(QColor(255, 255, 0, 80))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                    painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

                painter.end()

            if len(self.watershed_foreground_points) > 0:
                painter = QPainter(pixmap)
                pen = QPen(Qt.red)
                pen.setWidth(1)
                painter.setPen(pen)
                for point in self.watershed_foreground_points:
                    if point[0] == x_index:
                        painter.drawPoint(point[2], point[1])

                painter.end()
            self.yz_view.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 800, 800)
    window.setWindowTitle(f'3D TIFF Viewer')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

