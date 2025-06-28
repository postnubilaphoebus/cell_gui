from PyQt5.QtWidgets import (QGraphicsScene, 
                             QGraphicsPixmapItem, 
                             QGraphicsView, 
                             QMessageBox)
from cmaps import num_colors
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QTransform
import numpy as np

class GraphicsView(QGraphicsView):
    viewUpdated = pyqtSignal(QTransform)
    def __init__(self, main_window, view_plane, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.main_window = main_window
        self.view_plane = view_plane
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if self.view_plane == "XY":
            self.fixed_dim = "Z"
            self.missing_view_planes = ["XZ", "YZ"]
        elif self.view_plane == "XZ":
            self.fixed_dim = "Y"
            self.missing_view_planes = ["XY", "YZ"]
        elif self.view_plane == "YZ":
            self.fixed_dim = "X"
            self.missing_view_planes = ["XY", "XZ"]
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")

    def setPixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)

    def rotate_view(self, angle):
        self.rotate(angle)

    def focusInEvent(self, event):
        self.setStyleSheet("border: 1px solid lightgreen;")
        self.main_window.most_recent_focus = self.view_plane
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.setStyleSheet("border: 1px solid black;")
        super().focusOutEvent(event)

    def dragEnterEvent(self, event):
        # Let the MainWindow handle filedrops
        if event.mimeData().hasUrls():
            event.ignore()  
        else:
            event.ignore()  

    def dropEvent(self, event):
        # Let the MainWindow handle filedrops
        event.ignore()  

    def apply_transform(self, transform, mouse_position=None):
        self.setTransform(transform)
        if self.view_plane == "XY":
            v_scroll = self.main_window.xy_view_vertical_slider_val
            h_scroll = self.main_window.xy_view_horizontal_slider_val
        elif self.view_plane == "XZ":
            v_scroll = self.main_window.xz_view_vertical_slider_val
            h_scroll = self.main_window.xz_view_horizontal_slider_val
        elif self.view_plane == "YZ":
            v_scroll = self.main_window.yz_view_vertical_slider_val
            h_scroll = self.main_window.yz_view_horizontal_slider_val
        if h_scroll is not None:
            self.horizontalScrollBar().setValue(h_scroll)
        if v_scroll is not None:
            self.verticalScrollBar().setValue(v_scroll)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            current_value = self.main_window.slider.value()
            step_size = self.main_window.slider.singleStep()
            self.main_window.slider.setValue(current_value - step_size)
        elif event.key() == Qt.Key_2:
            current_value = self.main_window.slider.value()
            step_size = self.main_window.slider.singleStep()
            self.main_window.slider.setValue(current_value + step_size)
        elif event.key() == Qt.Key_3:
            current_value = self.main_window.slidery.value()
            step_size = self.main_window.slidery.singleStep()
            self.main_window.slidery.setValue(current_value - step_size)
        elif event.key() == Qt.Key_4:
            current_value = self.main_window.slidery.value()
            step_size = self.main_window.slidery.singleStep()
            self.main_window.slidery.setValue(current_value + step_size)
        elif event.key() == Qt.Key_5:
            current_value = self.main_window.sliderx.value()
            step_size = self.main_window.sliderx.singleStep()
            self.main_window.sliderx.setValue(current_value - step_size)
        elif event.key() == Qt.Key_6:
            current_value = self.main_window.sliderx.value()
            step_size = self.main_window.sliderx.singleStep()
            self.main_window.sliderx.setValue(current_value + step_size)
        elif event.key() == Qt.Key_M:
            self.main_window.markers_enabled = not self.main_window.markers_enabled
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        elif event.key() == Qt.Key_E:
            self.main_window.toggleEraser()
        elif event.key() == Qt.Key_A:
            self.main_window.toggleForeground()
        elif event.key() == Qt.Key_B:
            self.main_window.toggleBackground()
        elif event.key() == Qt.Key_V:
            self.main_window.hide_show_view_finder()
        elif event.key() == Qt.Key_Left:
            self.main_window.index_control.decrease_index()
            self.main_window.update_index_display()
        elif event.key() == Qt.Key_Right:
            self.main_window.index_control.increase_index()
            self.main_window.update_index_display()
        elif event.key() == Qt.Key_C:
            self.main_window.findCell()
        elif event.key() == Qt.Key_S:
            self.main_window.select_cell()
        elif event.key() == Qt.Key_D:
            self.main_window.delete_cell()
        elif event.key() == Qt.Key_P:
            self.main_window.switch_to_previous_tab()
        elif event.key() == Qt.Key_N:
            self.main_window.switch_to_next_tab()
        event.accept()

    def obtain_current_point(self, pixmap_item, event, view_plane):
        if view_plane == "XY":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            z_index = self.main_window.slider.value()
            return np.array([lp.x(), lp.y(), z_index])
        elif view_plane == "XZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            y_index = self.main_window.slidery.value()
            #return np.array([lp.x(), y_index, lp.y()])#
            return np.array([lp.y(), y_index, lp.x()])
        elif view_plane == "YZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            x_index = self.main_window.sliderx.value()
            return np.array([x_index, lp.y(), lp.x()])
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")
        
    def wheelEvent(self, event, recursion = True):
        if recursion:
            self.main_window.synchronize_wheeling(self.missing_view_planes, event)
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 0.8
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)
        self.centerOn(scene_pos)
        self.scale(factor, factor)
        delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
        self.centerOn(scene_pos - delta)
        event.accept()
        if self.view_plane == "XY":
            self.main_window.xy_transform = self.transform()
            self.main_window.xy_mouse_position = scene_pos
            self.main_window.xy_view_horizontal_slider_vale = self.horizontalScrollBar().value()
            self.main_window.xy_view_vertical_slider_vale = self.verticalScrollBar().value()
        elif self.view_plane == "XZ":
            self.main_window.xz_transform = self.transform()
            self.main_window.xz_mouse_position = scene_pos
            self.main_window.xz_view_horizontal_slider_vale = self.horizontalScrollBar().value()
            self.main_window.xz_view_vertical_slider_vale = self.verticalScrollBar().value()
        elif self.view_plane == "YZ":
            self.main_window.yz_transform = self.transform()
            self.main_window.yz_mouse_position = scene_pos
            self.main_window.yz_view_horizontal_slider_vale = self.horizontalScrollBar().value()
            self.main_window.yz_view_vertical_slider_vale = self.verticalScrollBar().value()

    def center_on_given_location(self, location):
        self.centerOn(location)

    def generate_nearby_points(self, center_point, fixed_dimension, distance):
        
        x, y, z = center_point
        dimension_ranges = {'X': (y, z), 'Y': (x, z), 'Z': (x, y)}
        bounds = {
            'X': (self.main_window.y_min, self.main_window.y_max, self.main_window.z_min, self.main_window.z_max),
            'Y': (self.main_window.x_min, self.main_window.x_max, self.main_window.z_min, self.main_window.z_max),
            'Z': (self.main_window.x_min, self.main_window.x_max, self.main_window.y_min, self.main_window.y_max),
        }
        points = []
        main_index = self.main_window.index_control.cell_index
        color_index = main_index % num_colors

        def in_bounds(v, min_v, max_v):
            return min_v <= v <= max_v

        def add_point(i, j, k):
            #if self.main_window.foreground_enabled:
            points.append((i, j, k, main_index, color_index))

        dim1, dim2 = dimension_ranges[fixed_dimension]
        min1, max1, min2, max2 = bounds[fixed_dimension]

        for v1 in range(dim1 - distance, dim1 + distance + 1):
            for v2 in range(dim2 - distance, dim2 + distance + 1):
                if (v1 - dim1) ** 2 + (v2 - dim2) ** 2 <= distance ** 2 and in_bounds(v1, min1, max1) and in_bounds(v2, min2, max2):
                    if fixed_dimension == 'X':
                        add_point(x, v1, v2)
                    elif fixed_dimension == 'Y':
                        add_point(v1, y, v2)
                    else:  # fixed_dimension == 'Z'
                        add_point(v1, v2, z)

        return points

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            if self.main_window.delete_cell_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_point(pixmap_item, event, self.view_plane)
                image = pixmap_item.pixmap().toImage()
                if self.view_plane == "XY":
                    x = point[0]
                    y = point[1]
                elif self.view_plane == "XZ":
                    x = point[0]
                    y = point[2]
                elif self.view_plane == "YZ":
                    x = point[2]
                    y = point[1]
                color = image.pixelColor(x, y)
                r, g, b, _ = color.getRgb()
                if not (r == g == b):
                    mbox = QMessageBox.question(self,  
                             "Delete Cell",  
                             "Are you sure you want to delete this cell?", 
                             QMessageBox.Yes | QMessageBox.No,  
                             QMessageBox.No)
                    if mbox == QMessageBox.Yes:
                        cell_idx = None
                        point = np.expand_dims(point, 0)
                        if self.view_plane == "XY":
                            z_index = self.main_window.slider.value()
                            view_points = self.main_window.z_view_dict.get(z_index)
                        elif self.view_plane == "XZ":
                            y_index = self.main_window.slidery.value()
                            view_points = self.main_window.y_view_dict.get(y_index)
                            point = point[:, [0, 2, 1]]
                        elif self.view_plane == "YZ":
                            x_index = self.main_window.sliderx.value()
                            view_points = self.main_window.x_view_dict.get(x_index)
                            point = point[:, [2, 1, 0]]
                        if view_points is not None:
                            point = point[:, :2]
                            matches = np.all(view_points[:, :2] == point, axis=1)
                            match_indices = np.where(matches)[0]
                            if match_indices.size > 0:
                                found_point = view_points[match_indices]
                                cell_idx = found_point[0][2]
                        if cell_idx is not None:
                            self.main_window.removeCell(cell_idx)
                event.accept()
                return

            if not self.main_window.new_cell_selected \
                and self.main_window.select_cell_enabled \
                and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_point(pixmap_item, event, self.view_plane)
                if point.size > 0:
                    cell_idx = None
                    point = np.expand_dims(point, 0)
                    if self.view_plane == "XY":
                        z_index = self.main_window.slider.value()
                        view_points = self.main_window.z_view_dict.get(z_index)
                    elif self.view_plane == "XZ":
                        y_index = self.main_window.slidery.value()
                        view_points = self.main_window.y_view_dict.get(y_index)
                        point = point[:, [0, 2, 1]]
                    elif self.view_plane == "YZ":
                        x_index = self.main_window.sliderx.value()
                        view_points = self.main_window.x_view_dict.get(x_index)
                        point = point[:, [2, 1, 0]]
                    if view_points is not None:
                        point = point[:, :2]
                        matches = np.all(view_points[:, :2] == point, axis=1)
                        match_indices = np.where(matches)[0]
                        if match_indices.size > 0:
                            found_point = view_points[match_indices]
                            cell_idx = found_point[0][2]
                    else:
                        return
                    if cell_idx:
                        self.main_window.new_cell_selected = True
                        self.main_window.index_control.cell_index = cell_idx
                        self.main_window.update_index_display()
                        self.main_window.alpha_label_index = cell_idx
                        self.main_window.index_control.update_index(cell_idx, 
                                                                    self.main_window.current_highest_cell_index)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
            else:
                self.main_window.dragging = True
                if self.main_window.drawing and self.main_window.markers_enabled:
                    pixmap_item = self._pixmap_item
                    points = self.obtain_current_point(pixmap_item, event, self.view_plane)
                    if self.main_window.foreground_enabled:
                        if self.main_window.brush_width == 1:
                            points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                            self.main_window.add_points(points)
                        elif self.main_window.brush_width > 1:
                            ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                            self.main_window.add_points(ppoints)
                    elif self.main_window.eraser_enabled:
                        if self.main_window.eraser_radius == 1:
                            points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                            self.main_window.removePoints(points, self.view_plane)
                        else:
                            ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.eraser_radius - 1)
                            self.main_window.removePoints(ppoints, self.view_plane)
                    self.main_window.update_xy_view()
                    self.main_window.update_xz_view()
                    self.main_window.update_yz_view()
                else:
                    self.main_window.last_mouse_pos = event.pos()

        elif event.button() == Qt.RightButton:

            if len(self.main_window.copied_points) > 0:
                # now shift by whatever index you are at
                origin_plane = self.main_window.copied_points[0][0]
                points_to_copy = self.main_window.copied_points[0][1]
                self.main_window.copied_points = []
                if origin_plane != self.view_plane:
                    return
                # now shift the points
                if self.view_plane == "XY":
                    z_index = self.main_window.slider.value()
                    points_to_copy[:, 2] = z_index
                elif self.view_plane == "XZ":
                    y_index = self.main_window.slidery.value()
                    points_to_copy[:, 1] = y_index
                elif self.view_plane == "YZ":
                    x_index = self.main_window.sliderx.value()
                    points_to_copy[:, 0] = x_index
                self.main_window.add_points(list(points_to_copy))
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
            else:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_point(pixmap_item, event, self.view_plane)
                if point.size > 0:
                    cell_idx = None
                    point = np.expand_dims(point, 0)
                    if self.view_plane == "XY":
                        z_index = self.main_window.slider.value()
                        view_points = self.main_window.z_view_dict.get(z_index)
                    elif self.view_plane == "XZ":
                        y_index = self.main_window.slidery.value()
                        view_points = self.main_window.y_view_dict.get(y_index)
                        point = point[:, [0, 2, 1]]
                    elif self.view_plane == "YZ":
                        x_index = self.main_window.sliderx.value()
                        view_points = self.main_window.x_view_dict.get(x_index)
                        point = point[:, [2, 1, 0]]
                    if view_points is not None:
                        point = point[:, :2]
                        matches = np.all(view_points[:, :2] == point, axis=1)
                        match_indices = np.where(matches)[0]
                        if match_indices.size > 0:
                            found_point = view_points[match_indices]
                            cell_idx = found_point[0][2]
                    else:
                        return
                    if cell_idx:
                        relevant_points = self.main_window.points_per_cell.get(cell_idx)
                        if self.view_plane == "XY":
                            points_to_copy = relevant_points[relevant_points[:, 2]==z_index]
                        elif self.view_plane == "XZ":
                            points_to_copy = relevant_points[relevant_points[:, 1]==y_index]
                        elif self.view_plane == "YZ":
                            points_to_copy = relevant_points[relevant_points[:, 0]==x_index]
                        if points_to_copy.size > 0:
                            self.main_window.copied_points.append((self.view_plane, points_to_copy))
        event.accept()

    def mouseMoveEvent(self, event):
        if self.main_window.drawing and self.main_window.dragging and self.main_window.markers_enabled:
            pixmap_item = self._pixmap_item
            points = self.obtain_current_point(pixmap_item, event, self.view_plane)
            if self.main_window.foreground_enabled:
                if self.main_window.brush_width == 1:
                    if points not in self.main_window.pure_coordinates:
                        points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                        self.main_window.add_points(points)
                elif self.main_window.brush_width > 1:
                    ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                    self.main_window.add_points(ppoints)
            elif self.main_window.eraser_enabled:
                if self.main_window.eraser_radius == 1:
                    points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                    self.main_window.removePoints(points, self.view_plane)
                else:
                    ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.eraser_radius - 1)
                    self.main_window.removePoints(ppoints, self.view_plane)
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        elif self.main_window.dragging:
            delta = event.pos() - self.main_window.last_mouse_pos
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.main_window.last_mouse_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.main_window.dragging = False
        event.accept()