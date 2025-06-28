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
from scipy.ndimage import label, find_objects
from cmaps import glasbey_cmap, glasbey_cmap_rgb
from gui_widgets import *
from graphics_view import GraphicsView
import imageio.v3 as iio
from skimage.measure import find_contours

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
        self.points_per_cell = {}
        self.copied_points = []
        self.relevant_xy_points = {}
        self.relevant_xz_points = {}
        self.relevant_yz_points = {}
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

        self.foreground_button = QPushButton('Annotate (A)', self)
        self.foreground_button.clicked.connect(self.toggleForeground)
        layout_buttons.addWidget(self.foreground_button)

        self.eraser_button = QPushButton('Eraser (E)', self)
        self.eraser_button.clicked.connect(self.toggleEraser)
        layout_buttons.addWidget(self.eraser_button)

        self.save_button = QPushButton('Save Masks', self)
        self.save_button.clicked.connect(self.save_file)
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

        self.select_cell_button = QPushButton('Select Cell (S)', self)
        self.select_cell_button.clicked.connect(self.select_cell)
        self.select_cell_enabled = False
        self.new_cell_selected = False
        layout_buttons.addWidget(self.select_cell_button)

        self.delete_cell_button = QPushButton('Delete Cell (D)', self)
        self.delete_cell_button.clicked.connect(self.delete_cell)
        layout_buttons.addWidget(self.delete_cell_button)
        self.delete_cell_enabled = False

        self.index_control = IndexControlWidget(self)
        layout_buttons.addWidget(self.index_control)
        self.index_control.increase_button.clicked.connect(self.update_index_display)
        self.index_control.decrease_button.clicked.connect(self.update_index_display)
        
        self.current_highest_cell_index = 0

        self.cell_idx_display = TextDisplay()
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)
        layout_buttons.addWidget(self.cell_idx_display)
        self.index_control.increase_button.clicked.connect(lambda: \
                                                           self.cell_idx_display.update_text(self.index_control.cell_index, 
                                                                                             self.current_highest_cell_index))
        self.index_control.decrease_button.clicked.connect(lambda: \
                                                           self.cell_idx_display.update_text(self.index_control.cell_index, 
                                                                                             self.current_highest_cell_index))

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
        self.alpha_label_index = None
        self.most_recent_focus = "XY"

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def update_highest_cell_index(self):
        if self.index_control.cell_index > self.current_highest_cell_index:
            self.current_highest_cell_index = self.index_control.cell_index

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

    def update_highest_cell_index(self):
        if self.index_control.cell_index > self.current_highest_cell_index:
            self.current_highest_cell_index = self.index_control.cell_index
            self.update_index_display()

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
                "copied_points": self.copied_points.copy(),
                "points_per_cell": {k: v.copy() for k, v in self.points_per_cell.items()}
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
                "copied_points": self.copied_points.copy(),
                "points_per_cell": {k: v.copy() for k, v in self.points_per_cell.items()}
            }
            self.tab_widget.setCurrentWidget(image_view_widget)

    def on_tab_changed(self, index):
        self.current_tab_index = index
        current_tab = self.tab_widget.currentWidget()
        self.alpha_label_index = None
        if self.data_per_tab.get(current_tab)is not None:
            self.update_tab_view(index)
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
            if self.most_recent_focus == "XY":
                self.xy_view.setFocus()
            elif self.most_recent_focus == "XZ":
                self.xz_view.setFocus()
            elif self.most_recent_focus == "YZ":
                self.yz_view.setFocus()
            else:
                self.xy_view.setFocus()

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
        self.points_per_cell = self.data_per_tab[current_tab].get("points_per_cell")
        self.copied_points = []
        self.synch_transform()
        
    # def numpyArrayToPixmap(self, img_np):
    #     img_np = np.require(img_np, np.uint8, 'C')
    #     if img_np.ndim == 3 and img_np.shape[2] == 3:
    #         qim = QImage(img_np.data, img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format_RGB888)
    #     else:
    #         qim = QImage(img_np.data, img_np.shape[1], 
    #                      img_np.shape[0], img_np.strides[0], 
    #                      QImage.Format_Indexed8)
    #     pixmap = QPixmap.fromImage(qim)
    #     return pixmap
    
    def numpyArrayToPixmap(self, arr: np.ndarray) -> QPixmap:
        arr = np.require(arr, np.uint8, 'C')
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c == 4:
                fmt = QImage.Format_RGBA8888
            elif c == 3:
                fmt = QImage.Format_RGB888
            else:
                raise ValueError("Unsupported number of channels")

            image = QImage(arr.data, w, h, arr.strides[0], fmt)
            return QPixmap.fromImage(image)
        else:
            if arr.ndim == 3 and arr.shape[2] == 3:
                qim = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
            else:
                qim = QImage(arr.data, arr.shape[1], 
                            arr.shape[0], arr.strides[0], 
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
                    self.load_masks(filename)
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
        if self.most_recent_focus == "XY":
            self.xy_view.setFocus()
        elif self.most_recent_focus == "XZ":
            self.xz_view.setFocus()
        elif self.most_recent_focus == "YZ":
            self.yz_view.setFocus()
        else:
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
            self.load_masks(file_name)
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
        print(f"Active Index: {index}, Item: {self.combo_box.currentText()}")
    
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
            self.alpha_label_index = None
            self.new_cell_selected = False
            #self.index_control.cell_index = self.current_highest_cell_index
            self.update_index_display()
            self.update_xy_view()
            self.update_xz_view()
            self.update_yz_view()
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
        if len(self.points_per_cell) > 0:
            self.current_highest_cell_index = max(self.points_per_cell.keys())
        self.cell_idx_display.update_text(self.index_control.cell_index, self.current_highest_cell_index)

    def toggleForeground(self):
        if not self.foreground_enabled:
            self.drawing = True
            self.foreground_enabled = True
            self.background_enabled = False
            self.eraser_enabled = False
            self.foreground_button.setStyleSheet("background-color: lightgreen")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.foreground_enabled = False
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
        if self.most_recent_focus == "XY":
            self.xy_view.setFocus()
        elif self.most_recent_focus == "XZ":
            self.xz_view.setFocus()
        elif self.most_recent_focus == "YZ":
            self.yz_view.setFocus()
        else:
            self.xy_view.setFocus()
        self.repaint()
        #self.central_widget.clearFocus()
        
    def toggleBackground(self):
        if not self.background_enabled:
            self.drawing = True
            self.foreground_enabled = False
            self.background_enabled = True
            self.eraser_enabled = False
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
            self.central_widget.setCursor(self.brush_cursor)
        else:
            self.drawing = False
            self.background_enabled = False
            self.foreground_button.setStyleSheet("")
            self.eraser_button.setStyleSheet("")
        self.repaint()
        #self.central_widget.clearFocus()
        
    def toggleEraser(self):
        self.foreground_enabled = False
        self.background_enabled = False
        if not self.eraser_enabled:
            self.eraser_enabled = True
            self.drawing = True
            self.eraser_button.setStyleSheet("background-color: lightgreen")
            self.foreground_button.setStyleSheet("")
        else:
            self.eraser_enabled = False
            self.drawing = False
            self.eraser_button.setStyleSheet("")
            self.foreground_button.setStyleSheet("")
        self.repaint()
        #self.central_widget.clearFocus()
        
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

    def remove_pts_slice_view_dicts(self, 
                                    temp_z_view_removal_dict, 
                                    temp_y_view_removal_dict, 
                                    temp_x_view_removal_dict):

        for k, v in temp_z_view_removal_dict.items():
            np_z_pts = self.z_view_dict.get(k)
            if np_z_pts is not None:
                coors_present = np_z_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = np.isin(coors_rows, v_rows)  
                indices_to_remove = np.argwhere(occupied_mask).flatten()
                if indices_to_remove.size == 0:
                    continue
                self.z_view_dict[k] = np.delete(np_z_pts, indices_to_remove, axis=0)
        for k, v in temp_y_view_removal_dict.items():
            np_y_pts = self.y_view_dict.get(k)
            if np_y_pts is not None:
                coors_present = np_y_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = np.isin(coors_rows, v_rows) 
                indices_to_remove = np.argwhere(occupied_mask).flatten()
                if indices_to_remove.size == 0:
                    continue
                self.y_view_dict[k] = np.delete(np_y_pts, indices_to_remove, axis=0)
        for k, v in temp_x_view_removal_dict.items():
            np_x_pts = self.x_view_dict.get(k)
            if np_x_pts is not None:
                coors_present = np_x_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = np.isin(coors_rows, v_rows) 
                indices_to_remove = np.argwhere(occupied_mask).flatten()
                if indices_to_remove.size == 0:
                    continue
                self.x_view_dict[k] = np.delete(np_x_pts, indices_to_remove, axis=0)

    def removeCell(self, cell_idx):
        # cell_points structure:
        # x, y, z, target_label, cell_id
        cell_points = self.points_per_cell.get(cell_idx)
        if cell_points is None:
            return
        temp_z_view_removal_dict = {}
        temp_y_view_removal_dict = {}
        temp_x_view_removal_dict = {}
        for p in cell_points:                   
            if p[2] not in temp_z_view_removal_dict:
                temp_z_view_removal_dict[p[2]] = []
            if p[1] not in temp_y_view_removal_dict:
                temp_y_view_removal_dict[p[1]] = []
            if p[0] not in temp_x_view_removal_dict:
                temp_x_view_removal_dict[p[0]] = []
            temp_z_view_removal_dict[p[2]].append((p[0], p[1], p[3], p[4]))
            temp_y_view_removal_dict[p[1]].append((p[2], p[0], p[3], p[4]))
            temp_x_view_removal_dict[p[0]].append((p[2], p[1], p[3], p[4]))
        temp_z_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_z_view_removal_dict.items()}
        temp_y_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_y_view_removal_dict.items()}
        temp_x_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_x_view_removal_dict.items()}
        self.remove_pts_slice_view_dicts(temp_z_view_removal_dict, temp_y_view_removal_dict, temp_x_view_removal_dict)
        self.points_per_cell.pop(cell_idx)
        self.update_xy_view()
        self.update_xz_view()
        self.update_yz_view()

        
    def removePoints(self, point, view_plane):
        if self.eraser_enabled and self.markers_enabled:
            # Determine the coordinate indices based on the view plane
            if view_plane == "XY":
                # Handle both single point and list of points
                if not isinstance(point, list) or len(point) == 5:
                    # Single point case
                    points_array = np.array([point])
                    z_plane_array = self.z_view_dict.get(point[2])
                else:
                    # Multiple points case
                    points_array = np.array(point)
                    z_plane_array = self.z_view_dict.get(points_array[0, 2])
                
                if z_plane_array is None:
                    return
                
                # Extract coordinates and target label (assuming all points have same target_label)
                eraser_x = points_array[:, 0]
                eraser_y = points_array[:, 1]
                target_label = points_array[0, 3]  # All points should have same target_label

                # Extract coordinates and labels from z_plane_array
                points_x = z_plane_array[:, 0]
                points_y = z_plane_array[:, 1]
                points_labels = z_plane_array[:, 3]

                # Vectorized conditions - check if any eraser point is within radius
                label_match = points_labels == target_label

                points_x_match = np.isin(points_x, eraser_x)
                points_y_match = np.isin(points_y, eraser_y) 
                mask = label_match & points_x_match & points_y_match 

                # Build points_to_remove as numpy array using vectorized indexing
                if np.any(mask):
                    selected_points = z_plane_array[mask]
                    # For multiple eraser points, we use the first point's index (point[2]) 
                    # as they should all have the same z-coordinate/index
                    point_z_index = points_array[0, 2]
                    points_to_remove = np.column_stack([
                        selected_points[:, 0],  # p[0] - x coordinate
                        selected_points[:, 1],  # p[1] - y coordinate
                        np.full(len(selected_points), point_z_index),  # point[2] - z index
                        selected_points[:, 2],  # p[2] - label
                        selected_points[:, 3]   # p[3] - color index
                    ])
                else:
                    points_to_remove = []
            elif view_plane == "XZ":
                # Handle both single point and list of points
                if not isinstance(point, list) or len(point) == 5:
                    # Single point case
                    points_array = np.array([point])
                    y_plane_array = self.y_view_dict.get(point[1])
                else:
                    # Multiple points case
                    points_array = np.array(point)
                    y_plane_array = self.y_view_dict.get(points_array[0, 1])
                if y_plane_array is None:
                    return

                # Extract coordinates and target label (assuming all points have same target_label)
                eraser_x = points_array[:, 0]
                eraser_y = points_array[:, 2]
                target_label = points_array[0, 3]  # All points should have same target_label
                
                # Extract coordinates and labels from z_plane_array
                points_x = y_plane_array[:, 1]
                points_y = y_plane_array[:, 0]
                points_labels = y_plane_array[:, 3]

                # Vectorized conditions - check if any eraser point is within radius
                label_match = points_labels == target_label
                points_x_match = np.isin(points_x, eraser_x) 
                points_y_match = np.isin(points_y, eraser_y) 

                # Combine all conditions
                mask = label_match & points_x_match & points_y_match 

                # Build points_to_remove as numpy array using vectorized indexing
                if np.any(mask):
                    selected_points = y_plane_array[mask]
                    # For multiple eraser points, we use the first point's index (point[2]) 
                    # as they should all have the same z-coordinate/index
                    point_y_index = points_array[0, 1]
                    points_to_remove = np.column_stack([
                        selected_points[:, 1],  # p[0] - x coordinate
                        np.full(len(selected_points), point_y_index),
                        selected_points[:, 0],  # p[1] - z coordinate
                        selected_points[:, 2],  # p[2] - label
                        selected_points[:, 3]   # p[3] - color index
                    ])
                else:
                    points_to_remove = []
            elif view_plane == "YZ":
                # Handle both single point and list of points
                if not isinstance(point, list) or len(point) == 5:
                    # Single point case
                    points_array = np.array([point])
                    x_plane_array = self.x_view_dict.get(point[0])
                else:
                    # Multiple points case
                    points_array = np.array(point)
                    x_plane_array = self.x_view_dict.get(points_array[0, 0])
                if x_plane_array is None:
                    return
                # Extract coordinates and target label (assuming all points have same target_label)
                eraser_x = points_array[:, 2]
                eraser_y = points_array[:, 1]
                target_label = points_array[0, 3]  # All points should have same target_label

                # Extract coordinates and labels from x_plane_array
                points_x = x_plane_array[:, 0]
                points_y = x_plane_array[:, 1]
                points_labels = x_plane_array[:, 3]

                # Vectorized conditions - check if any eraser point is within radius
                label_match = points_labels == target_label

                points_x_match = np.isin(points_x, eraser_x) 
                points_y_match = np.isin(points_y, eraser_y) 
                mask = label_match & points_x_match & points_y_match 

                # Build points_to_remove as numpy array using vectorized indexing
                if np.any(mask):
                    selected_points = x_plane_array[mask]
                    # For multiple eraser points, we use the first point's index (point[2]) 
                    # as they should all have the same z-coordinate/index
                    point_x_index = points_array[0, 0]
                    points_to_remove = np.column_stack([
                        np.full(len(selected_points), point_x_index),
                        selected_points[:, 1],  # p[0] - y coordinate
                        selected_points[:, 0],  # p[1] - z coordinate
                        selected_points[:, 2],  # p[2] - label
                        selected_points[:, 3]   # p[3] - color index
                    ])
                else:
                    points_to_remove = []
            else:
                return
            
            if len(points_to_remove) == 0:
                return

            temp_z_view_removal_dict = {}
            temp_y_view_removal_dict = {}
            temp_x_view_removal_dict = {}
            for p in points_to_remove:                   
                if p[2] not in temp_z_view_removal_dict:
                    temp_z_view_removal_dict[p[2]] = []
                if p[1] not in temp_y_view_removal_dict:
                    temp_y_view_removal_dict[p[1]] = []
                if p[0] not in temp_x_view_removal_dict:
                    temp_x_view_removal_dict[p[0]] = []
                temp_z_view_removal_dict[p[2]].append((p[0], p[1], p[3], p[4]))
                temp_y_view_removal_dict[p[1]].append((p[2], p[0], p[3], p[4]))
                temp_x_view_removal_dict[p[0]].append((p[2], p[1], p[3], p[4]))
            temp_z_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_z_view_removal_dict.items()}
            temp_y_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_y_view_removal_dict.items()}
            temp_x_view_removal_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_x_view_removal_dict.items()}
            self.remove_pts_slice_view_dicts(temp_z_view_removal_dict, temp_y_view_removal_dict, temp_x_view_removal_dict)
            points_of_target_label = self.points_per_cell.get(target_label)
            mask_to_remove = (points_to_remove[None, :, :] == points_of_target_label[:, None, :]).all(-1).any(1)
            new_points_of_target_label = points_of_target_label[~mask_to_remove]
            self.points_per_cell[target_label] = new_points_of_target_label

        
    def load_masks(self, filename):
        # Show the loading screen
        self.loading_screen = LoadingScreen()
        self.loading_screen.show()
        # Create and start the worker thread
        self.mask_loader = MaskLoader(self, filename)
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

    def add_pts_slice_view_dicts(self, 
                                 temp_z_view_dict, 
                                 temp_y_view_dict, 
                                 temp_x_view_dict,
                                 temp_points_per_cell_dict):
        for k, v in temp_z_view_dict.items():
            np_z_pts = self.z_view_dict.get(k)
            if np_z_pts is not None:
                coors_present = np_z_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = ~np.isin(v_rows, coors_rows)
                v = v[occupied_mask]
                if v.size == 0:
                    continue
                self.z_view_dict[k] = np.append(np_z_pts, v, axis=0)
            else:
                self.z_view_dict[k] = v
        for k, v in temp_y_view_dict.items():
            np_y_pts = self.y_view_dict.get(k)
            if np_y_pts is not None:
                coors_present = np_y_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = ~np.isin(v_rows, coors_rows)
                v = v[occupied_mask]
                if v.size == 0:
                    continue
                self.y_view_dict[k] = np.append(np_y_pts, v, axis=0)
            else:
                self.y_view_dict[k] = v
        for k, v in temp_x_view_dict.items():
            np_x_pts = self.x_view_dict.get(k)
            if np_x_pts is not None:
                coors_present = np_x_pts[:, :2]
                v_rows = v[:, :2].view([('', v.dtype)] * 2).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 2).reshape(-1)
                occupied_mask = ~np.isin(v_rows, coors_rows)
                v = v[occupied_mask]
                if v.size == 0:
                    continue
                self.x_view_dict[k] = np.append(np_x_pts, v, axis=0)
            else:
                self.x_view_dict[k] = v
        for k, v in temp_points_per_cell_dict.items():
            np_pts = self.points_per_cell.get(k)
            if np_pts is not None:
                coors_present = np_pts[:, :3]
                v_rows = v[:, :3].view([('', v.dtype)] * 3).reshape(-1)
                coors_rows = coors_present.view([('', coors_present.dtype)] * 3).reshape(-1)
                occupied_mask = ~np.isin(v_rows, coors_rows)
                v = v[occupied_mask]
                if v.size == 0:
                    continue
                self.points_per_cell[k] = np.append(np_pts, v, axis=0)
            else:
                self.points_per_cell[k] = v

    def add_points(self, point):
        if not isinstance(point, list):
            point = [point]
        temp_z_view_dict = {}
        temp_y_view_dict = {}
        temp_x_view_dict = {}
        temp_points_per_cell_dict = {}
        for p in point:                   
            #self.foreground_points.append(p)
            if p[3] not in temp_points_per_cell_dict:
                temp_points_per_cell_dict[p[3]] = []
            if p[2] not in temp_z_view_dict:
                temp_z_view_dict[p[2]] = []
            if p[1] not in temp_y_view_dict:
                temp_y_view_dict[p[1]] = []
            if p[0] not in temp_x_view_dict:
                temp_x_view_dict[p[0]] = []
            temp_points_per_cell_dict[p[3]].append(p)
            temp_z_view_dict[p[2]].append((p[0], p[1], p[3], p[4]))
            temp_y_view_dict[p[1]].append((p[2], p[0], p[3], p[4]))
            temp_x_view_dict[p[0]].append((p[2], p[1], p[3], p[4]))
        temp_z_view_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_z_view_dict.items()}
        temp_y_view_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_y_view_dict.items()}
        temp_x_view_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_x_view_dict.items()}
        temp_points_per_cell_dict = {k: np.array(v, dtype=np.int32) for k, v in temp_points_per_cell_dict.items()}
        self.add_pts_slice_view_dicts(temp_z_view_dict, temp_y_view_dict, temp_x_view_dict, temp_points_per_cell_dict)

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
            self.load_masks(file_name)

    def findCell(self):
        cell_points = self.points_per_cell.get(self.current_highest_cell_index)
        if cell_points is not None:
            xyz_points = cell_points[:, :3]
            average_point = np.median(xyz_points, axis=0)
            self.slidery.setValue(round(average_point[1]))
            self.sliderx.setValue(round(average_point[0]))
            self.slider.setValue(round(average_point[2]))
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
                    if self.alpha_label_index is not None:
                        mask_for_contour_finding = np.where(label_indices == self.alpha_label_index)[0]
                        pts_relevant = xy[mask_for_contour_finding]
                        if pts_relevant.shape[0] > 0:
                            max_box = np.max(pts_relevant, axis=0)
                            min_box = np.min(pts_relevant, axis=0)
                    colors = glasbey_cmap_rgb[label_indices]  
                    points_to_paint = np.hstack((xy, colors))
                    gray_rgb = np.stack([image]*3, axis=-1)  
                    gray_rgb[points_to_paint[:, 1], points_to_paint[:, 0]] = points_to_paint[:, 2:]
                    pixmap = self.numpyArrayToPixmap(gray_rgb)
                    if self.alpha_label_index is not None and pts_relevant.shape[0] > 0:
                        circle_painter = QPainter(pixmap)
                        pen = QPen(QColor(255, 0, 0, 80))
                        pen.setWidth(1)
                        circle_painter.setPen(pen)
                        circle_painter.drawEllipse(min_box[0] - 2, min_box[1] - 2, max_box[0] - min_box[0] + 4, max_box[1] - min_box[1] + 4)
                        circle_painter.end()
                else:
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
                    if self.alpha_label_index is not None:
                        mask_for_contour_finding = np.where(label_indices == self.alpha_label_index)[0]
                        pts_relevant = xz[mask_for_contour_finding]
                        if pts_relevant.shape[0] > 0:
                            max_box = np.max(pts_relevant, axis=0)
                            min_box = np.min(pts_relevant, axis=0)
                    if self.alpha_label_index is not None and pts_relevant.shape[0] > 0:
                        circle_painter = QPainter(pixmap)
                        pen = QPen(QColor(255, 0, 0, 80))
                        pen.setWidth(1)
                        circle_painter.setPen(pen)
                        circle_painter.drawEllipse(min_box[0] - 2, min_box[1] - 2, max_box[0] - min_box[0] + 4, max_box[1] - min_box[1] + 4)
                        circle_painter.end()
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
                    gray_rgb = np.stack([image]*3, axis=-1)
                    gray_rgb[points_to_paint[:, 1], points_to_paint[:, 0]] = points_to_paint[:, 2:]
                    pixmap = self.numpyArrayToPixmap(gray_rgb)
                    if self.alpha_label_index is not None:
                        mask_for_contour_finding = np.where(label_indices == self.alpha_label_index)[0]
                        pts_relevant = yz[mask_for_contour_finding]
                        if pts_relevant.shape[0] > 0:
                            max_box = np.max(pts_relevant, axis=0)
                            min_box = np.min(pts_relevant, axis=0)
                    if self.alpha_label_index is not None and pts_relevant.shape[0] > 0:
                        circle_painter = QPainter(pixmap)
                        pen = QPen(QColor(255, 0, 0, 80))
                        pen.setWidth(1)
                        circle_painter.setPen(pen)
                        circle_painter.drawEllipse(min_box[0] - 2, min_box[1] - 2, max_box[0] - min_box[0] + 4, max_box[1] - min_box[1] + 4)
                        circle_painter.end()
                else:
                    pixmap = self.numpyArrayToPixmap(image)
                
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
    window.setWindowTitle(f'3D Image Stack Editor and Viewer')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
