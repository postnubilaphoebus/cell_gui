from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

from colormap import num_colors


class GraphicsView(QGraphicsView):
    def __init__(self, main_window, view_plane, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
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
        elif self.view_plane == "XZ":
            self.fixed_dim = "Y"
        elif self.view_plane == "YZ":
            self.fixed_dim = "X"
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")

    def setPixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)

    def rotate_view(self, angle):
        self.rotate(angle)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            current_value = self.main_window.sliderz.value()
            step_size = self.main_window.sliderz.singleStep()
            self.main_window.sliderz.setValue(current_value - step_size)
        elif event.key() == Qt.Key_2:
            current_value = self.main_window.sliderz.value()
            step_size = self.main_window.sliderz.singleStep()
            self.main_window.sliderz.setValue(current_value + step_size)
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
            if self.main_window.visualization_only:
                if self.main_window.markers_enabled:
                    if self.main_window.backup_greyscale is not None:
                        self.main_window.backup_color = self.main_window.image_data
                        self.main_window.image_data = self.main_window.backup_greyscale
                else:
                    if self.main_window.backup_color is not None:
                        self.main_window.backup_greyscale = self.main_window.image_data
                        self.main_window.image_data = self.main_window.backup_color
            self.main_window.markers_enabled = not self.main_window.markers_enabled
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        elif event.key() == Qt.Key_E:
            self.main_window.toggleEraser()
        elif event.key() == Qt.Key_J:
            self.main_window.visualization_mode()
        elif event.key() == Qt.Key_F:
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
        elif event.key() == Qt.Key_L:
            self.main_window.localContrastEnhancer()
        elif event.key() == Qt.Key_O:
            self.main_window.open_file_dialog()
        event.accept()

    def obtain_current_points(self, pixmap_item, event, view_plane):
        if view_plane == "XY":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            z_index = self.main_window.sliderz.value()
            return (lp.x(), lp.y(), z_index)
        elif view_plane == "XZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            y_index = self.main_window.slidery.value()
            return (lp.y(), y_index, lp.x())
        elif view_plane == "YZ":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            x_index = self.main_window.sliderx.value()
            return (x_index, lp.y(), lp.x())
        else:
            raise ValueError("Invalid viewplane.\
                             Choose among 'XY', 'XZ', 'YZ'")

    def wheelEvent(self, event):
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

    def generate_points_around(self, center_point, fixed_dimension, distance):
        points = []
        x, y, z = center_point
        if fixed_dimension == 'X':
            for j in range(y - distance, y + distance + 1):
                for k in range(z - distance, z + distance + 1):
                    if (j - y) ** 2 + (k - z) ** 2 <= distance ** 2 \
                            and j <= self.main_window.y_max \
                            and j >= self.main_window.y_min \
                            and k <= self.main_window.z_max \
                            and k >= self.main_window.z_min:
                        if self.main_window.foreground_enabled:
                            points.append((x, j, k, self.main_window.index_control.cell_index,
                                           self.main_window.index_control.cell_index % num_colors))
                        else:
                            points.append((x, j, k))
        elif fixed_dimension == 'Y':
            for i in range(x - distance, x + distance + 1):
                for k in range(z - distance, z + distance + 1):
                    if (i - x) ** 2 + (k - z) ** 2 <= distance ** 2 \
                            and i <= self.main_window.x_max \
                            and i >= self.main_window.x_min \
                            and k <= self.main_window.z_max \
                            and k >= self.main_window.z_min:
                        if self.main_window.foreground_enabled:
                            points.append((i, y, k, self.main_window.index_control.cell_index,
                                           self.main_window.index_control.cell_index % num_colors))
                        else:
                            points.append((i, y, k))
        elif fixed_dimension == 'Z':
            for i in range(x - distance, x + distance + 1):
                for j in range(y - distance, y + distance + 1):
                    if (i - x) ** 2 + (j - y) ** 2 <= distance ** 2 \
                            and i <= self.main_window.x_max \
                            and i >= self.main_window.x_min \
                            and j <= self.main_window.y_max \
                            and j >= self.main_window.y_min:
                        if self.main_window.foreground_enabled:
                            points.append((i, j, z, self.main_window.index_control.cell_index,
                                           self.main_window.index_control.cell_index % num_colors))
                        else:
                            points.append((i, j, z))
        return points

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            if not self.main_window.new_cell_selected \
                    and self.main_window.select_cell_enabled \
                    and self.main_window.foreground_points \
                    and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if point:
                    cell_idx = None
                    for p in self.main_window.foreground_points:
                        if tuple(p[:3]) == point:
                            cell_idx = p[3]
                            break
                    if cell_idx:
                        self.main_window.new_cell_selected = True
                        self.main_window.index_control.cell_index = cell_idx
                        self.main_window.update_index_display()
                        self.main_window.index_control.update_index(self.main_window.index_control.cell_index,
                                                                    self.main_window.current_highest_cell_index)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
            else:
                self.main_window.dragging = True
                if self.main_window.local_contrast_enhancer_enabled:
                    pixmap_item = self._pixmap_item
                    sp = self.mapToScene(event.pos())
                    lp = pixmap_item.mapFromScene(sp).toPoint()
                    self.main_window.first_mouse_pos_for_contrast_rect = lp
                else:
                    if self.main_window.drawing and self.main_window.markers_enabled:
                        pixmap_item = self._pixmap_item
                        points = self.obtain_current_points(pixmap_item, event, self.view_plane)
                        if self.main_window.foreground_enabled:
                            if self.main_window.brush_width == 1:
                                points = points + (self.main_window.index_control.cell_index,) + (
                                    self.main_window.index_control.cell_index % num_colors,)
                                if points not in self.main_window.foreground_points:
                                    self.main_window.foreground_points.append(points)
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_points_around(points, self.fixed_dim,
                                                                      self.main_window.brush_width - 1)
                                for pp in ppoints:
                                    if pp not in self.main_window.foreground_points:
                                        self.main_window.foreground_points.append(pp)
                        elif self.main_window.background_enabled:
                            if self.main_window.brush_width == 1:
                                if points not in self.main_window.background_points:
                                    self.main_window.background_points.append(points)
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_points_around(points, self.fixed_dim,
                                                                      self.main_window.brush_width - 1)
                                for pp in ppoints:
                                    if pp not in self.main_window.background_points:
                                        self.main_window.background_points.append(pp)
                        elif self.main_window.eraser_enabled:
                            self.main_window.removePoints(points, self.view_plane)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
                    else:
                        self.main_window.last_mouse_pos = event.pos()

        elif event.button() == Qt.RightButton:

            if self.main_window.copied_points:
                view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                slider_value = self.main_window.sliderz.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                for p in self.main_window.copied_points:
                    if p[-1] == view_plane_index:
                        new_point = list(p[:4])
                        new_point[view_plane_index] = slider_value
                        self.main_window.foreground_points.append(tuple(new_point))
                self.main_window.copied_points = []
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
            else:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if point:
                    cell_idx = None
                    view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                    slider_value = self.main_window.sliderz.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                    for p in self.main_window.foreground_points:
                        if tuple(p[:3]) == point:
                            cell_idx = p[3]
                            break
                    if cell_idx:
                        for p in self.main_window.foreground_points:
                            if p[3] == cell_idx and p[view_plane_index] == slider_value:
                                self.main_window.copied_points.append(p + (view_plane_index,))
        event.accept()

    def mouseMoveEvent(self, event):

        if self.main_window.local_contrast_enhancer_enabled and self.main_window.dragging:
            pixmap_item = self._pixmap_item
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            self.main_window.last_mouse_pos_for_contrast_rect = lp
            self.main_window.update_xy_view()
            self.main_window.update_xz_view()
            self.main_window.update_yz_view()
        else:
            if self.main_window.drawing and self.main_window.dragging and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                points = self.obtain_current_points(pixmap_item, event, self.view_plane)
                if self.main_window.foreground_enabled:
                    if self.main_window.brush_width == 1:
                        points = points + (self.main_window.index_control.cell_index,) + (
                            self.main_window.index_control.cell_index % num_colors,)
                        if points not in self.main_window.foreground_points:
                            self.main_window.foreground_points.append(points)
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                        for pp in ppoints:
                            if pp not in self.main_window.foreground_points:
                                self.main_window.foreground_points.append(pp)
                elif self.main_window.background_enabled:
                    if self.main_window.brush_width == 1:
                        if points not in self.main_window.background_points:
                            self.main_window.background_points.append(points)
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_points_around(points, self.fixed_dim, self.main_window.brush_width - 1)
                        for pp in ppoints:
                            if pp not in self.main_window.background_points:
                                self.main_window.background_points.append(pp)
                elif self.main_window.eraser_enabled:
                    self.main_window.removePoints(points, self.view_plane)
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
