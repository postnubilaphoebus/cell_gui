import sys
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QVBoxLayout, 
                             QWidget, 
                             QSlider, 
                             QPushButton, 
                             QLineEdit, 
                             QLabel,  
                             QMessageBox, 
                             QFileDialog,
                             QSplitter,
                             QAction,
                             QTabWidget)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
import matplotlib.ticker as mticker
import math
from scipy.ndimage import label, find_objects
from cmaps import glasbey_cmap, glasbey_cmap_rgb
from gui_widgets import *
from graphics_view import GraphicsView
import imageio.v3 as iio
from numba import njit


@njit
def fast_indexing(indices, colormap):
  out = np.empty((indices.shape[0], 3), dtype=np.uint8)
  for i in range(indices.shape[0]):
    out[i] = colormap[indices[i]]
  return out

class MainWindow(QMainWindow):
    label_shift_answer = pyqtSignal(bool)
    def __init__(self, filename = None):
        super().__init__()
        self.setAcceptDrops(True)
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
        self.xy_view_horizontal_slider_val = None
        self.xz_view_horizontal_slider_val = None
        self.yz_view_horizontal_slider_val = None
        self.xy_view_vertical_slider_val = None
        self.xz_view_vertical_slider_val = None
        self.yz_view_vertical_slider_val = None
        self.current_zoom_location = None#scene_pos - delta
        self.current_zoom_factor = 1

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
        save_action.setShortcut("Ctrl+S")
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

        self.tab_switch_prev = QPushButton("Go to previous tab (P)", self)
        self.tab_switch_prev.clicked.connect(self.switch_to_previous_tab)
        layout_buttons.addWidget(self.tab_switch_prev)

        self.tab_switch_next = QPushButton("Go to next tab (N)", self)
        self.tab_switch_next.clicked.connect(self.switch_to_next_tab)
        layout_buttons.addWidget(self.tab_switch_next)
        
        self.brush_label = QLabel("Brush Width:")
        layout_buttons.addWidget(self.brush_label)
        self.cursor_pix = QPixmap('paintbrush_icon.png')
        self.cursor_scaled_pix = self.cursor_pix.scaled(QSize(50, 50), Qt.KeepAspectRatio)
        self.brush_cursor = QCursor(Qt.ArrowCursor)

        self.droplet_cursor_pix = QPixmap('droplet_cursor.png')
        self.droplet_cursor_scaled_pix = self.droplet_cursor_pix.scaled(QSize(20, 20), Qt.KeepAspectRatio)
        
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

        self.foreground_button = QPushButton('Brush (B)', self)
        self.foreground_button.clicked.connect(self.toggleForeground)
        layout_buttons.addWidget(self.foreground_button)

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

        self.background_threshold = 0.0
        self.cell_centre_threshold = 0.0
        self.markers_off_on_button = QPushButton('Masks Off/On (M)', self)
        self.markers_off_on_button.clicked.connect(self.markersOffOn)
        layout_buttons.addWidget(self.markers_off_on_button)

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
        self.slider.valueChanged.connect(self.update_slider_text)
        self.slidery.valueChanged.connect(self.update_xz_view)
        self.slidery.valueChanged.connect(self.update_slidery_text)
        self.sliderx.valueChanged.connect(self.update_yz_view)
        self.slidery.valueChanged.connect(self.update_xy_view)
        self.sliderx.valueChanged.connect(self.update_xy_view)
        self.slidery.valueChanged.connect(self.update_yz_view)
        self.sliderx.valueChanged.connect(self.update_xz_view)
        self.sliderx.valueChanged.connect(self.update_sliderx_text)

        self.side_widget = QWidget()
        self.side_layout = QVBoxLayout(self.side_widget)
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
        self.eraser_enabled = False
        self.last_mouse_pos = None
        self.first_mouse_pos_for_contrast_rect = None
        self.last_mouse_pos_for_contrast_rect = None
        self.first_mouse_pos_for_watershed_cube = None
        self.last_mouse_pos_for_watershed_cube = None
        self.xy_transform = None
        self.xz_transform = None
        self.yz_transform = None
        self.xy_mouse_position = None
        self.xz_mouse_position = None
        self.yz_mouse_position = None
        self.foreground_points = []
        self.background_points = []
        self.z_view_dict = {}
        self.y_view_dict = {}
        self.x_view_dict = {}
        self.pure_coordinates = []
        self.copied_points = []
        self.relevant_xy_points = {}
        self.relevant_xz_points = {}
        self.relevant_yz_points = {}
        self.markers_enabled = True
        self.update_xz_view()
        self.update_yz_view()
        self.update_xy_view()
        self.xy_painter = None
        self.xz_painter = None
        self.yz_painter = None
        self.relevant_xy_points_loaded = False
        self.relevant_xz_points_loaded = False
        self.relevant_yz_points_loaded = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def slider_value_text(self, val):
        return f"Z-Planes (1,2): {val}/{self.slider.maximum()}"
    
    def update_slider_text(self):
        self.slider_label.setText(self.slider_value_text(self.slider.value()))
    
    def slidery_value_text(self, val):
        return f"Y-Planes (3,4): {val}/{self.slidery.maximum()}"
    
    def update_slidery_text(self):
        self.slidery_label.setText(self.slidery_value_text(self.slidery.value()))
    
    def sliderx_value_text(self, val):
        return f"X-Planes (5,6): {val}/{self.sliderx.maximum()}"
    
    def update_sliderx_text(self):
        self.sliderx_label.setText(self.sliderx_value_text(self.sliderx.value()))

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)

    def switch_to_previous_tab(self):
        current_index = self.tab_widget.currentIndex()
        num_tabs = self.tab_widget.count()
        previous_index = (current_index - 1) % num_tabs
        previous_tab_widget = self.tab_widget.widget(previous_index)
        if self.data_per_tab.get(previous_tab_widget) is not None:
            self.current_tab_index = previous_index
            self.tab_widget.setCurrentIndex(previous_index)
            self.update_tab_view(previous_index)
            # self.update_xy_view()
            # self.update_xz_view()
            # self.update_yz_view()

    def switch_to_next_tab(self):
        current_index = self.tab_widget.currentIndex()
        num_tabs = self.tab_widget.count()
        next_index = (current_index + 1) % num_tabs
        next_tab_widget = self.tab_widget.widget(next_index)
        if self.data_per_tab.get(next_tab_widget) is not None:
            self.current_tab_index = next_index
            self.tab_widget.setCurrentIndex(next_index)
            self.update_tab_view(next_index)
            # self.update_xy_view()
            # self.update_xz_view()
            # self.update_yz_view()

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

            xy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            xy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            xy_box = QHBoxLayout()
            xz_box = QHBoxLayout()
            yz_box = QHBoxLayout()

            pixmapxy = QPixmap()
            pixmapxy.fill(Qt.white)
            pixmapxz = QPixmap()
            pixmapxz.fill(Qt.white)
            pixmapyz = QPixmap()
            pixmapyz.fill(Qt.white)

            xy_view.setPixmap(pixmapxy)
            xz_view.setPixmap(pixmapxz)
            yz_view.setPixmap(pixmapyz)

            xy_box.addWidget(xy_view)
            xz_box.addWidget(xz_view)
            yz_box.addWidget(yz_view)

            label = QLabel("xy view")
            label.setAlignment(Qt.AlignCenter)
            xy_box.addWidget(label)

            label = QLabel("xz view")
            label.setAlignment(Qt.AlignCenter)
            xz_box.addWidget(label)

            label = QLabel("yz view")
            label.setAlignment(Qt.AlignCenter)
            yz_box.addWidget(label)

            layout.addLayout(xy_box, stretch=1)
            layout.addLayout(xz_box, stretch=1)
            layout.addLayout(yz_box, stretch=1)

            # Add the widget to the tab widget
            self.tab_widget.addTab(image_view_widget, image_name)
            self.xy_view = xy_view
            self.xz_view = xz_view
            self.yz_view = yz_view
            self.initial_view = 1
            self.tab_widget.setCurrentWidget(image_view_widget)

        elif self.initial_view == 1:
            old_tab = self.tab_widget.widget(0)
            if old_tab is not None:
                self.tab_widget.removeTab(0)
                old_tab.deleteLater() 

            image_view_widget = QWidget()
            layout = QVBoxLayout(image_view_widget)

            xy_view = GraphicsView(self, "XY")
            #xy_view = GraphicsViewVispy(self, "XY")
            xz_view = GraphicsView(self, "XZ")
            yz_view = GraphicsView(self, "YZ")

            yz_view.rotate_view(-90)
            xz_view.rotate_view(-90)

            

            xy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            xy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            xy_box = QHBoxLayout()
            xz_box = QHBoxLayout()
            yz_box = QHBoxLayout()

            xy_box.addWidget(xy_view)
            xz_box.addWidget(xz_view)
            yz_box.addWidget(yz_view)

            label = QLabel("xy view")
            label.setAlignment(Qt.AlignCenter)
            xy_box.addWidget(label)

            label = QLabel("xz view")
            label.setAlignment(Qt.AlignCenter)
            xz_box.addWidget(label)

            label = QLabel("yz view")
            label.setAlignment(Qt.AlignCenter)
            yz_box.addWidget(label)

            layout.addLayout(xy_box, stretch=1)
            layout.addLayout(xz_box, stretch=1)
            layout.addLayout(yz_box, stretch=1)

            self.tab_widget.addTab(image_view_widget, image_name)

            self.xy_view = xy_view
            self.xz_view = xz_view
            self.yz_view = yz_view

            self.xy_view_horizontal_slider_val = self.xy_view.horizontalScrollBar().value()
            self.xy_view_vertical_slider_val = self.xy_view.verticalScrollBar().value()
            self.xz_view_horizontal_slider_val = self.xz_view.horizontalScrollBar().value()
            self.xz_view_vertical_slider_val = self.xz_view.verticalScrollBar().value()
            self.yz_view_horizontal_slider_val = self.yz_view.horizontalScrollBar().value()
            self.yz_view_vertical_slider_val = self.yz_view.verticalScrollBar().value()


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
            self.filename = image_name
            self.current_highest_cell_index = 0
            self.foreground_points = []
            self.background_points = []
            self.z_view_dict = {}
            self.y_view_dict = {}
            self.x_view_dict = {}
            self.pure_coordinates = []
            self.copied_points = []


            
            # layout.addWidget(xy_view, stretch=1)
            # #layout.addWidget(xy_view.get_qt_widget(), stretch=1)
            # layout.addWidget(xz_view, stretch=1)
            # layout.addWidget(yz_view, stretch=1)

            #self.tab_widget.insertTab(0, image_view_widget, image_name)
            self.tab_widget.setCurrentWidget(image_view_widget)
            current_tab = self.tab_widget.currentWidget()
            if current_tab not in self.data_per_tab:
                self.data_per_tab[current_tab] = {}
            self.data_per_tab[current_tab] = {
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
                "z_view_dict": {k: v.copy() for k, v in self.z_view_dict.items()},#self.z_view_dict.copy(),
                "y_view_dict": {k: v.copy() for k, v in self.y_view_dict.items()},#self.y_view_dict.copy(),
                "x_view_dict": {k: v.copy() for k, v in self.x_view_dict.items()},# self.x_view_dict.copy(),
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
            xy_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            xy_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            xz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            yz_view.rotate_view(-90)
            xz_view.rotate_view(-90)
            xy_box = QHBoxLayout()
            xz_box = QHBoxLayout()
            yz_box = QHBoxLayout()
            
            xy_box.addWidget(xy_view)
            xz_box.addWidget(xz_view)
            yz_box.addWidget(yz_view)
            label = QLabel("xy view")
            label.setAlignment(Qt.AlignCenter)
            xy_box.addWidget(label)

            label = QLabel("xz view")
            label.setAlignment(Qt.AlignCenter)
            xz_box.addWidget(label)

            label = QLabel("yz view")
            label.setAlignment(Qt.AlignCenter)
            yz_box.addWidget(label)
            layout.addLayout(xy_box, stretch=1)
            layout.addLayout(xz_box, stretch=1)
            layout.addLayout(yz_box, stretch=1)

            self.xy_view_horizontal_slider_val = self.xy_view.horizontalScrollBar().value()
            self.xy_view_vertical_slider_val = self.xy_view.verticalScrollBar().value()
            self.xz_view_horizontal_slider_val = self.xz_view.horizontalScrollBar().value()
            self.xz_view_vertical_slider_val = self.xz_view.verticalScrollBar().value()
            self.yz_view_horizontal_slider_val = self.yz_view.horizontalScrollBar().value()
            self.yz_view_vertical_slider_val = self.yz_view.verticalScrollBar().value()

            # Add the widget to the tab widget
            #self.tab_widget.addTab(image_view_widget, image_name)
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

            # layout.addWidget(xy_view, stretch=1)
            # layout.addWidget(xz_view, stretch=1)
            # layout.addWidget(yz_view, stretch=1)

            self.tab_widget.addTab(image_view_widget, image_name)
            self.tab_widget.setCurrentWidget(image_view_widget)
            current_tab = self.tab_widget.currentWidget()
            if current_tab not in self.data_per_tab:
                self.data_per_tab[current_tab] = {}
            self.data_per_tab[current_tab] = {
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
                "z_view_dict": {k: v.copy() for k, v in self.z_view_dict.items()},#self.z_view_dict.copy(),
                "y_view_dict": {k: v.copy() for k, v in self.y_view_dict.items()},#self.y_view_dict.copy(),
                "x_view_dict": {k: v.copy() for k, v in self.x_view_dict.items()},#self.x_view_dict.copy(),
                "pure_coordinates": self.pure_coordinates.copy(),
                "copied_points": self.copied_points.copy()
            }
            self.tab_widget.setCurrentWidget(image_view_widget)

    def on_tab_changed(self, index):
        self.current_tab_index = index
        current_tab = self.tab_widget.currentWidget()
        if self.data_per_tab.get(current_tab)is not None:
            self.update_tab_view(index)
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            self.xy_view.setFocus() # always focus on xy view by default

    def close_tab(self, index):
        num_tabs = self.tab_widget.count()
        if num_tabs > 1:
            del self.data_per_tab[self.tab_widget.widget(index)]
            self.tab_widget.removeTab(index)

    def synch_transform(self):
        if self.xy_transform is not None:
            if self.xy_mouse_position is None:
                self.xy_view.apply_transform(self.xy_transform)
            else:
                self.xy_view.apply_transform(self.xy_transform, self.xy_mouse_position)
        if self.xz_transform is not None:
            if self.xz_mouse_position is None:
                self.xz_view.apply_transform(self.xz_transform)
            else:
                self.xz_view.apply_transform(self.xz_transform, self.xz_mouse_position)
        if self.yz_transform is not None:
            if self.yz_mouse_position is None:
                self.yz_view.apply_transform(self.yz_transform)
            else:
                self.yz_view.apply_transform(self.yz_transform, self.yz_mouse_position)

            
    def update_tab_view(self, index):
        current_tab = self.tab_widget.currentWidget()
        self.image_min = self.data_per_tab[current_tab].get("image_min")
        self.image_max = self.data_per_tab[current_tab].get("image_max")
        self.z_max = self.data_per_tab[current_tab].get("z_max")
        self.y_max = self.data_per_tab[current_tab].get("y_max")
        self.x_max = self.data_per_tab[current_tab].get("x_max")
        self.z_min = self.data_per_tab[current_tab].get("z_min")
        self.y_min = self.data_per_tab[current_tab].get("y_min")
        self.x_min = self.data_per_tab[current_tab].get("x_min")
        self.min_pixel_intensity = self.data_per_tab[current_tab].get("min_pixel_intensity")
        self.max_pixel_intensity = self.data_per_tab[current_tab].get("max_pixel_intensity")
        self.image_data = self.data_per_tab[current_tab].get("image_data")
        self.brush_width = self.data_per_tab[current_tab].get("brush_width")
        self.eraser_radius = self.data_per_tab[current_tab].get("eraser_radius")
        self.num_channels = self.data_per_tab[current_tab].get("num_channels")
        self.xy_view = self.data_per_tab[current_tab].get("xy_view")
        self.xz_view = self.data_per_tab[current_tab].get("xz_view")
        self.yz_view = self.data_per_tab[current_tab].get("yz_view")
        self.filename = self.data_per_tab[current_tab].get("filename")
        self.current_highest_cell_current_tab = self.data_per_tab[current_tab].get("current_highest_cell_current_tab")
        self.foreground_points = self.data_per_tab[current_tab].get("foreground_points")
        self.background_points = self.data_per_tab[current_tab].get("background_points")
        self.z_view_dict = self.data_per_tab[current_tab].get("z_view_dict")
        self.y_view_dict = self.data_per_tab[current_tab].get("y_view_dict")
        self.x_view_dict = self.data_per_tab[current_tab].get("x_view_dict")
        self.pure_coordinates = self.data_per_tab[current_tab].get("pure_coordinates")
        self.copied_points = []
        self.synch_transform()
        
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

    def imagej_auto_contrast(self, image, saturated=0.35):
        image = image.astype(np.float32)
        flat = image.flatten()
        n_pixels = len(flat)
        saturated_pixel_count = int(n_pixels * saturated / 100.0)
        saturated_pixel_count = min(saturated_pixel_count, n_pixels // 2 - 1)  # avoid index overflow
        sorted_pixels = np.sort(flat)
        # Handle very small saturation count
        if saturated_pixel_count == 0:
            min_val = sorted_pixels[0]
            max_val = sorted_pixels[-1]
        else:
            min_val = sorted_pixels[saturated_pixel_count]
            max_val = sorted_pixels[-saturated_pixel_count - 1]
        # Prevent division by zero
        if max_val == min_val:
            return np.clip(image, 0, 1) 
        # Stretch contrast
        stretched = (image - min_val) / (max_val - min_val)
        stretched = np.clip(stretched, 0, 1)

        return stretched

    def load_image(self, filename):
        # Load image data
        self.filename = filename
        try:
            ext = os.path.splitext(filename)[1].lower()
            mbox = QMessageBox(self)
            mbox.setWindowTitle('Mask Or Image?')
            mbox.setText("Do you want to load a mask or an image?")
            load_mask_btn = mbox.addButton("Load Mask", QMessageBox.RejectRole)
            load_image_btn = mbox.addButton("Load Image", QMessageBox.AcceptRole)
            mbox.exec_()
            if mbox.clickedButton() == load_mask_btn:
                if self.tab_widget.count() >= 1:
                    self.load_masks(filename, True)
                    return None
                else:
                    QMessageBox.critical(self, "Error", "Failed to load mask as no image present:")
                    return None
            elif mbox.clickedButton() == load_image_btn:
                if ext == '.npy':
                    image_data = np.load(filename)
                else:
                    image_data = iio.imread(filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return None

        shape = image_data.shape
        if self.initial_view > 1:
            previous_tab = self.tab_widget.widget(0)
            current_shape = self.data_per_tab[previous_tab]["image_data"].shape
            if shape != current_shape:
                QMessageBox.warning(self, "Invalid Image",
                                    "Image dimensions do not match the current image." \
                                    "This GUI is made to load images of the same shape" \
                                    " concurrently")
                return None
        
        if len(shape) == 3:
            if shape[-1] in [3, 4] and np.issubdtype(image_data.dtype, np.integer):
                QMessageBox.warning(self, "Invalid Image",
                                    "The image appears to be a 2D color image, but a greyscale 3D volume is required.")
                return None
        elif len(shape) == 4:
            # Shape might be (T, Z, Y, X) or (Z, Y, X, C)
            if shape[-1] in [3, 4] and np.issubdtype(image_data.dtype, np.integer):
                QMessageBox.warning(self, "Invalid Image",
                                    "The image appears to be a 3D color stack, but a single-channel 3D volume is required.")
                return None
        elif len(shape) < 3:
            QMessageBox.warning(self, "Invalid Image",
                                "Image must be at least 3D.")
            return None
        image_data = self.imagej_auto_contrast(image_data)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        image_data = (image_data * 255.0).astype(np.uint8)
        self.filename_list.append(filename)

        # Update sliders with new image dimensions
        if self.initial_view <= 1:
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

        # Update image parameters and views
        self.image_min = 0
        self.image_max = 255
        self.min_pixel_intensity = np.min(self.image_data.ravel())
        self.max_pixel_intensity = np.max(self.image_data.ravel())
        self.update_xy_view()
        self.update_yz_view()
        self.update_xz_view()

        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)
        self.synch_transform()
        self.xy_view.setFocus()
    
    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.tif *.jpg *.png);;NumPy Files (*.npy);;All Files (*)",
            options=options
        )
        if not file_name:
            return
        self.load_image(file_name)

    def open_mask(self):
        if self.image_data is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "Numpy Files (*.npy)", options=options)
            if not file_name:
                return
            self.load_masks(file_name, False)
        else:
            QMessageBox.information(self, "No Image Found", "Please load an image first!")

    def save_file(self):
        mask = np.zeros_like(self.image_data, dtype=np.int32) 
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File')
        if not filename:
            return
        while not (filename.endswith(".npy") or filename.endswith(".tif") or filename.endswith(".tiff")):
            QMessageBox.warning(self, "Invalid File Name", "File name should end with .npy, .tif, or .tiff.")
            filename, _ = QFileDialog.getSaveFileName(self, 'Save File')
            if not filename:
                return

        z_dimension = mask.shape[0]
        for i in range(z_dimension):
            xy_points = self.z_view_dict.get(i)
            if xy_points is None:
                continue
            xy_coors = xy_points[:, :2]
            mask[i, xy_coors[:, 1], xy_coors[:, 0]] = xy_points[:, 2]

        if filename.endswith(".npy"):
            np.save(filename, mask)
        elif filename.endswith(".tif") or filename.endswith(".tiff"):
            iio.imwrite(filename, mask)
    
    def on_selection_change(self, index):
        print(f"Selected index: {index}, Item: {self.combo_box.currentText()}")
    
    def synchronize_wheeling(self, missing_view_planes, wheel_event):
        graphics_views = [self.xy_view, self.xz_view, self.yz_view]
        for view in graphics_views:
            if view.view_plane in missing_view_planes:
                view.wheelEvent(wheel_event, False)

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

    @pyqtSlot(int)
    def handle_progress(self, progress):
        self.progress_label.setVisible(True)
        self.progress_label.setText(f"Progress: {progress:.1f}%")

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
            import time
            if view_plane == "XY":
                t0 = time.perf_counter()
                z_plane_points = self.z_view_dict.get(point[2])
                if z_plane_points is None:
                    return
                # points_to_remove = points[label_mask][close_mask]
                # point_1_column = np.full((points_to_remove.shape[0], 1), point[1])
                # points_to_remove = np.hstack((points_to_remove[:, :1], point_1_column, points_to_remove[:, 1:]))
                #points_to_remove = [(p[0], p[1], point[2], p[2], p[3]) for p in z_plane_points if (p[3] == point[4] and math.dist((point[0], point[1]), (p[0], p[1])) <= self.eraser_radius)]
                radius = self.eraser_radius
                px, py = point[0], point[1]
                target_label = point[4]

                points_to_remove = [
                (p[0], p[1], point[2], p[2], p[3])
                for p in z_plane_points
                if (p[3] == target_label and
                    abs(p[0] - px) <= radius and
                    abs(p[1] - py) <= radius and
                    math.dist((px, py), (p[0], p[1])) <= radius)
                ]
                #print("first points removed")
                t1 = time.perf_counter()
            elif view_plane == "XZ":
                y_plane_points = self.y_view_dict.get(point[1])
                if y_plane_points is None:
                    return
                radius = self.eraser_radius
                px, pz = point[0], point[2]
                target_label = point[4]

                points_to_remove = [
                (p[1], point[1], p[0], p[2], p[3])
                for p in y_plane_points
                if (p[3] == target_label and
                    abs(p[0] - px) <= radius and
                    abs(p[1] - pz) <= radius and
                    math.dist((px, pz), (p[1], p[0])) <= radius)
                ]
                #print("second points removed")
                t2 = time.perf_counter()
                #points_to_remove = [(p[1], point[1], p[0], p[2], p[3]) for p in y_plane_points if (p[3] == point[4] and math.dist((point[0], point[2]), (p[1], p[0])) <= self.eraser_radius)]
            elif view_plane == "YZ":
                x_plane_points = self.x_view_dict.get(point[0])
                if x_plane_points is None:
                    return
                radius = self.eraser_radius
                py, pz = point[1], point[2]
                target_label = point[4]

                points_to_remove = [
                (point[0], p[1], p[0], p[2], p[3])
                for p in x_plane_points
                if (p[3] == target_label and
                    abs(p[0] - py) <= radius and
                    abs(p[1] - pz) <= radius and
                    math.dist((py, pz), (p[1], p[0])) <= radius)
                ]
                #print("third points removed")
                t3 = time.perf_counter()
                #points_to_remove = [(point[0], p[1], p[0], p[2], p[3]) for p in x_plane_points if (p[3] == point[4] and math.dist((point[1], point[2]), (p[1], p[0])) <= self.eraser_radius)]
                points_to_remove = [(point[0], p[1], p[0], p[2], p[3]) for p in x_plane_points if (p[3] == point[4] and math.dist((point[1], point[2]), (p[1], p[0])) <= self.eraser_radius)]
            else:
                return  # Invalid view_plane, nothing to remove
            
            points_to_remove = np.array(points_to_remove)
            t1 = time.perf_counter()
            
            pure_points_to_remove = points_to_remove[:, :3]
            #pure_points_to_remove = points_to_remove[:, :3]
            t2 = time.perf_counter()


            # self.foreground_points = [
            #     p for p in self.foreground_points if p not in points_to_remove
            # ]
            foreground_points = self.foreground_points

            mask = ~np.isin(foreground_points, points_to_remove)

            self.foreground_points = foreground_points[mask]

            points_to_remove_set = set(points_to_remove)
            self.foreground_points = [
                p for p in self.foreground_points if p not in points_to_remove_set
            ]


            # self.background_points = [
            #     p for p in self.background_points if p not in points_to_remove
            # ]

            t3 = time.perf_counter()

            
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
            t4 = time.perf_counter()

            total_time = t4 - t0
            first_time = (t1 - t0) / total_time
            second_time = (t2 - t1) / total_time
            third_time = (t3 - t2) / total_time
            fourth_time = (t4 - t3) / total_time
            print(f"First: {first_time}, Second: {second_time}, Third: {third_time}, Fourth: {fourth_time}")


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
        self.mask_loader.error_signal.connect(self.show_error)
        self.mask_loader.ask_user_signal.connect(self.shift_minimum_index)
        self.label_shift_answer.connect(self.mask_loader.on_user_answer)
        self.mask_loader.finished.connect(self.on_masks_loaded)
        self.mask_loader.start()

    @pyqtSlot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Shape Error", message)

    @pyqtSlot()
    def shift_minimum_index(self):
        answer = QMessageBox.question(self, f'Warning: lowest index is larger than 0', 
                                    "Only zero is considered background. Do you want to shift the lowest index to 0?", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        answer_bool = (answer == QMessageBox.Yes)
        self.label_shift_answer.emit(answer_bool)

    def add_points(self, point):
        if not isinstance(point, list):
            point = [point]
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

    def findCell(self):
        pts = self.foreground_points
        if pts:
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
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        else:
            self.markers_enabled = True
            self.markers_off_on_button.setText("Markers On (M)")
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
        self.repaint()

    def update_xy_view(self):
        if self.image_data is not None:
            z_index = self.slider.value()
            image = self.get_flat_image_view("XY", z_index)
            if self.markers_enabled:
                relevant_points = self.z_view_dict.get(z_index)
                if relevant_points is not None:
                    points_to_paint = []
                    xy = relevant_points[:, :2]
                    label_indices = relevant_points[:, -1]
                    colors = glasbey_cmap_rgb[label_indices]  
                    points_to_paint = np.hstack((xy, colors))
                    gray_rgb = np.stack([image]*3, axis=-1)  # shape (H, W, 3)
                    gray_rgb[points_to_paint[:, 1], points_to_paint[:, 0]] = points_to_paint[:, 2:]
                    #self.xy_view.setPixmap(QPixmap.fromImage(QImage(gray_rgb, gray_rgb.shape[1], gray_rgb.shape[0], QImage.Format_RGB888)))

                    # total_time = t2-t0
                    # first_time = (t1-t0)/total_time
                    # second_time = (t2-t1)/total_time
                    # print("first_time", first_time)
                    # print("second_time", second_time)
                    # print("total_time", total_time)

                    #points_to_paint = generate_points_array(relevant_points, glasbey_cmap_rgb)
                    # for point in relevant_points:
                    #     points_to_paint.append((point[0], point[1], glasbey_cmap_rgb[point[-1]]))
                    #t3 = time.perf_counter()
                    # gray_rgb = np.stack([image]*3, axis=-1)
                    # for x, y, rgb in points_to_paint:
                    #     gray_rgb[y, x] = rgb  # Note: NumPy is (row=y, col=x)
                    #t4 = time.perf_counter()
                    # gray_rgb = np.stack([image]*3, axis=-1)  # shape (H, W, 3)
                    # xy = points_to_paint[:, :2].astype(int)     # shape (N, 2)
                    # colors = points_to_paint[:, 2:].astype(gray_rgb.dtype)  # shape (N, 3)
                    # gray_rgb[xy[:,1], xy[:,0]] = colors
                    #t1 = time.perf_counter()
                    pixmap = self.numpyArrayToPixmap(gray_rgb)
                    #t2 = time.perf_counter()

                    # t5 = time.perf_counter()
                else:
                    # t1 = time.perf_counter()
                    # t2 = time.perf_counter()
                    pixmap = self.numpyArrayToPixmap(image)

                if self.view_finder:
                    self.xy_painter = QPainter(pixmap)
                    pen = QPen(QColor(255, 255, 0, 80))
                    pen.setWidth(1)
                    self.xy_painter.setPen(pen)
                    y_val = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                    x_val = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.width())
                    square_size = 30 
                    top_left_x = x_val - square_size // 2
                    top_left_y = y_val - square_size // 2
                    self.xy_painter.drawEllipse(top_left_x, top_left_y, square_size, square_size)
                    self.xy_painter.drawLine(x_val, 0, x_val, pixmap.height())
                    self.xy_painter.drawLine(0, y_val, pixmap.width(), y_val)
                    self.xy_painter.end()

            else:
                pixmap = self.numpyArrayToPixmap(image)

            # pixmap = self.numpyArrayToPixmap(image)
            # t1 = time.perf_counter()
            # if self.markers_enabled:
            #     self.xy_painter = QPainter(pixmap)
            #     if self.foreground_points:
            #         color_idx = 0
            #         color_count = 0
            #         # pen = QPen(QColor(glasbey_cmap[color_idx]))
            #         # pen.setWidth(1)
            #         # self.xy_painter.setPen(pen)
            #         t2 = time.perf_counter()
            #         relevant_points = self.z_view_dict.get(z_index)
            #         t3 = time.perf_counter()

            #         # h, w = image.shape[:2]
            #         # #canvas = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
            #         # canvas = np.repeat(np.expand_dims(image, -1), 4, axis=-1).astype(np.uint8)
            #         # for p in relevant_points:
            #         #     x, y, class_id = p[0], p[1], p[-1]
            #         #     color = glasbey_cmap_rgb[class_id]
            #         #     canvas[y, x] = [*color[:3], 255]  # Set pixel

            #         # qimage = QImage(canvas.data, w, h, QImage.Format_RGBA8888)
            #         # pixmap = QPixmap.fromImage(qimage)
            #         if relevant_points:
            #             for point in relevant_points:
            #                 if point[-1] != color_idx:
            #                     color_idx = point[-1]
            #                     color_count += 1
            #                     pen = QPen(QColor(glasbey_cmap[color_idx]))
            #                     pen.setWidth(1)
            #                     self.xy_painter.setPen(pen)
            #                 self.xy_painter.drawPoint(point[0], point[1])
            #         # t4 = time.perf_counter()

            #         # total_time = t4 - t0
            #         # first_time_fraction = (t1 - t0) / total_time
            #         # second_time_fraction = (t2 - t1) / total_time
            #         # third_time_fraction = (t3 - t2) / total_time
            #         # fourth_time_fraction = (t4 - t3) / total_time
            #         # print("first_time_fraction", first_time_fraction)
            #         # print("second_time_fraction", second_time_fraction)
            #         # print("third_time_fraction", third_time_fraction)
            #         # print("fourth_time_fraction", fourth_time_fraction)
            #         # print("total_time", total_time)

                

            self.xy_view.setPixmap(pixmap)
            # t3 = time.perf_counter()
            # total_time = t3 - t0
            # first_time_fraction = (t1 - t0) / total_time
            # second_time_fraction = (t2 - t1) / total_time
            # third_time_fraction = (t3 - t2) / total_time
            # print("first_time_fraction", first_time_fraction)
            # print("second_time_fraction", second_time_fraction)
            # print("third_time_fraction", third_time_fraction)
            # print("total_time", total_time)
            

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
            #pixmap = self.numpyArrayToPixmap(image)

            if self.markers_enabled:
                relevant_points = self.y_view_dict.get(y_index)
                if relevant_points is not None:
                    points_to_paint = []
                    xz = relevant_points[:, :2]
                    label_indices = relevant_points[:, -1]
                    colors = glasbey_cmap_rgb[label_indices]  
                    points_to_paint = np.hstack((xz, colors))
                    gray_rgb = np.stack([image]*3, axis=-1)  # shape (H, W, 3)
                    gray_rgb[points_to_paint[:, 1], points_to_paint[:, 0]] = points_to_paint[:, 2:]

                    

                    pixmap = self.numpyArrayToPixmap(gray_rgb)

                    

                    # if self.view_finder:
                    #     painter = QPainter(pixmap)
                    #     pen = QPen(QColor(255, 255, 0, 80))
                    #     pen.setWidth(1)
                    #     painter.setPen(pen)
                    #     y_val = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                    #     x_val = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.width())
                    #     square_size = 30 
                    #     top_left_x = x_val - square_size // 2
                    #     top_left_y = y_val - square_size // 2
                    #     painter.drawEllipse(top_left_x, top_left_y, square_size, square_size)
                    #     painter.drawLine(x_val, 0, x_val, pixmap.height())
                    #     painter.drawLine(0, y_val, pixmap.width(), y_val)

                    #     painter.end()

                    

                #painter = QPainter(pixmap)
                # if self.foreground_points:
                #     color_idx = 0
                #     color_count = 0
                #     pen = QPen(QColor(glasbey_cmap[color_idx]))
                #     pen.setWidth(1)
                #     painter.setPen(pen)
                #     #relevant_points = [point for point in self.foreground_points if point[1] == y_index]
                #     relevant_points = self.y_view_dict.get(y_index)
                #     if relevant_points is not None:
                #         for point in relevant_points:
                #             if point[-1] != color_idx:
                #                 color_idx = point[-1]
                #                 color_count += 1
                #                 pen = QPen(QColor(glasbey_cmap[color_idx]))
                #                 pen.setWidth(1)
                #                 painter.setPen(pen)
                #             painter.drawPoint(point[0], point[1])


                # pen = QPen(Qt.blue)
                # pen.setWidth(1)
                # painter.setPen(pen)
                else:
                    pixmap = self.numpyArrayToPixmap(image)

                if self.view_finder:
                    painter = QPainter(pixmap)
                    pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                    pixmapy = self.slider_to_pixmap(self.sliderx.value(), 0, self.x_max, 0, pixmap.height())
                    pen = QPen(QColor(255, 255, 0, 80))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                    painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

                    painter.end()

            else:
                pixmap = self.numpyArrayToPixmap(image)

            self.xz_view.setPixmap(pixmap)

    def update_yz_view(self):
        if self.image_data is not None:
            x_index = self.sliderx.value()
            image = self.get_flat_image_view("YZ", x_index)
            pixmap = self.numpyArrayToPixmap(image)
            if self.markers_enabled:
                relevant_points = self.x_view_dict.get(x_index)
                if relevant_points is not None:
                    points_to_paint = []
                    yz = relevant_points[:, :2]
                    label_indices = relevant_points[:, -1]
                    colors = glasbey_cmap_rgb[label_indices]  
                    points_to_paint = np.hstack((yz, colors))
                    gray_rgb = np.stack([image]*3, axis=-1)  # shape (H, W, 3)
                    gray_rgb[points_to_paint[:, 1], points_to_paint[:, 0]] = points_to_paint[:, 2:]
                    pixmap = self.numpyArrayToPixmap(gray_rgb)
                else:
                    pixmap = self.numpyArrayToPixmap(image)
                # painter = QPainter(pixmap)
                # if self.foreground_points:
                #     color_idx = 0
                #     color_count = 0
                #     pen = QPen(QColor(glasbey_cmap[color_idx]))
                #     pen.setWidth(1)
                #     painter.setPen(pen)
                #     #relevant_points = [point for point in self.foreground_points if point[0] == x_index]
                #     relevant_points = self.x_view_dict.get(x_index)
                #     if relevant_points is not None:
                #         for point in relevant_points:
                #             if point[-1] != color_idx:
                #                 color_idx = point[-1]
                #                 color_count += 1
                #                 pen = QPen(QColor(glasbey_cmap[color_idx]))
                #                 pen.setWidth(1)
                #                 painter.setPen(pen)
                #             painter.drawPoint(point[0], point[1])

                # pen = QPen(Qt.blue)
                # pen.setWidth(1)
                # painter.setPen(pen)

                if self.view_finder:    
                    painter = QPainter(pixmap)
                    pixmapy = self.slider_to_pixmap(self.slidery.value(), 0, self.y_max, 0, pixmap.height())
                    pixmapx = self.slider_to_pixmap(self.slider.value(), 0, self.z_max, 0, pixmap.width())
                    pen = QPen(QColor(255, 255, 0, 80))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawLine(pixmapx, 0, pixmapx, pixmap.height())
                    painter.drawLine(0, pixmapy, pixmap.width(), pixmapy)

                    painter.end()

            else:
                pixmap = self.numpyArrayToPixmap(image)

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