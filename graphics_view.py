from PyQt5.QtWidgets import (QGraphicsScene, 
                             QGraphicsPixmapItem, 
                             QGraphicsView, 
                             QMessageBox)
from cmaps import num_colors
from PyQt5.QtCore import Qt

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
        elif event.key() == Qt.Key_D:
            self.main_window.delete_cell()
        elif event.key() == Qt.Key_X:
            self.main_window.labelCorrections()
        elif event.key() == Qt.Key_Plus:
            if self.main_window.watershed_seeding_point and self.main_window.watershed_neighborhood_selection_enabled:
                self.main_window.watershed_radius += 2
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
        elif event.key() == Qt.Key_Minus:
            if self.main_window.watershed_seeding_point and self.main_window.watershed_neighborhood_selection_enabled:
                if self.main_window.watershed_radius > 1:
                    self.main_window.watershed_radius -= 2
                    self.main_window.update_xy_view()
                    self.main_window.update_xz_view()
                    self.main_window.update_yz_view()
        elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.main_window.apply_watershed()
        event.accept()

    def obtain_current_point(self, pixmap_item, event, view_plane):
        import time
        if view_plane == "XY":
            sp = self.mapToScene(event.pos())
            lp = pixmap_item.mapFromScene(sp).toPoint()
            z_index = self.main_window.slider.value()
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
        self.main_window.current_zoom_location = scene_pos - delta
        event.accept()

    def center_on_given_location(self, location):
        self.centerOn(location)

    #def synchronize_wheeling_between_tabs()

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
            if self.main_window.foreground_enabled:
                points.append((i, j, k, main_index, color_index))
            else:
                points.append((i, j, k))

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
                if point:
                    mbox = QMessageBox.question(self,  
                             "Delete Cell",  
                             "Are you sure you want to delete this cell?", 
                             QMessageBox.Yes | QMessageBox.No,  
                             QMessageBox.No)
                    if mbox == QMessageBox.Yes:
                        cell_idx = None
                        for p in self.main_window.foreground_points:
                            if tuple(p[:3]) == point:
                                cell_idx = p[3]
                                break
                        indices_to_remove = []
                        for i, p in enumerate(self.main_window.foreground_points):
                            if p[3] == cell_idx:
                                indices_to_remove.append(i)
                        self.main_window.remove_indices(indices_to_remove)
                event.accept()
                return
            
            if self.main_window.watershed_neighborhood_selection_enabled:
                pixmap_item = self._pixmap_item
                self.main_window.watershed_seeding_point = self.obtain_current_point(pixmap_item, event, self.view_plane)
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()

            if not self.main_window.new_cell_selected \
                and self.main_window.select_cell_enabled \
                and self.main_window.foreground_points \
                and self.main_window.markers_enabled:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_point(pixmap_item, event, self.view_plane)
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
                        points = self.obtain_current_point(pixmap_item, event, self.view_plane)

                        if self.main_window.foreground_enabled:
                            if self.main_window.brush_width == 1:
                                if points not in self.main_window.pure_coordinates:
                                    points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                                    self.main_window.add_points(points)
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                                relevant_points = [pp for pp in ppoints if pp[:3] not in self.main_window.pure_coordinates]
                                self.main_window.add_points(relevant_points)
                                # for pp in ppoints:
                                #     if pp[:3] not in self.main_window.pure_coordinates:
                                #         self.main_window.add_points(pp)
                        elif self.main_window.background_enabled:
                            if self.main_window.brush_width == 1:
                                if points not in self.main_window.pure_coordinates:
                                    self.main_window.add_points(points, category = "background")
                            elif self.main_window.brush_width > 1:
                                ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                                for pp in ppoints:
                                    if pp[:3] not in self.main_window.pure_coordinates:
                                        self.main_window.add_points(pp[:3], category = "background")
                        elif self.main_window.eraser_enabled:
                            points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                            self.main_window.removePoints(points, self.view_plane)
                        self.main_window.update_xy_view()
                        self.main_window.update_xz_view()
                        self.main_window.update_yz_view()
                    else:
                        self.main_window.last_mouse_pos = event.pos()

        elif event.button() == Qt.RightButton:

            if self.main_window.watershed_neighborhood_selection_enabled:
                self.main_window.watershed_seeding_point = None
                self.main_window.watershed_radius = 2
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()

            if self.main_window.copied_points:
                view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                slider_value = self.main_window.slider.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                for p in self.main_window.copied_points:
                    if p[-1] == view_plane_index:
                        new_point = list(p[:5])
                        new_point[view_plane_index] = slider_value
                        self.main_window.add_points(tuple(new_point))
                        #self.main_window.foreground_points.append(tuple(new_point))
                self.main_window.copied_points = []
                self.main_window.update_xy_view()
                self.main_window.update_xz_view()
                self.main_window.update_yz_view()
            else:
                pixmap_item = self._pixmap_item
                point = self.obtain_current_point(pixmap_item, event, self.view_plane)
                if point:
                    cell_idx = None
                    view_plane_index = 2 if self.view_plane == 'XY' else 1 if self.view_plane == 'XZ' else 0
                    slider_value = self.main_window.slider.value() if view_plane_index == 2 else self.main_window.slidery.value() if view_plane_index == 1 else self.main_window.sliderx.value()
                    for p in self.main_window.foreground_points:
                        if tuple(p[:3]) == point:
                            cell_idx = p[3]
                            break
                    if cell_idx:
                        for p in self.main_window.foreground_points:
                            if p[3] == cell_idx and p[view_plane_index] == slider_value:
                                self.main_window.copied_points.append(p + (p[-1],) + (view_plane_index,))
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
                points = self.obtain_current_point(pixmap_item, event, self.view_plane)
                if self.main_window.foreground_enabled:
                    if self.main_window.brush_width == 1:
                        if points not in self.main_window.pure_coordinates:
                            points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
                            self.main_window.add_points(points)
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                        relevant_points = [pp for pp in ppoints if pp[:3] not in self.main_window.pure_coordinates]
                        self.main_window.add_points(relevant_points)
                elif self.main_window.background_enabled:
                    if self.main_window.brush_width == 1:
                        if points not in self.main_window.pure_coordinates:
                            self.main_window.add_points(points, category = "background")
                    elif self.main_window.brush_width > 1:
                        ppoints = self.generate_nearby_points(points, self.fixed_dim, self.main_window.brush_width - 1)
                        for pp in ppoints:
                            if pp[:3] not in self.main_window.pure_coordinates:
                                self.main_window.add_points(pp[:3], category = "background")
                elif self.main_window.eraser_enabled:
                    points = points + (self.main_window.index_control.cell_index,) + (self.main_window.index_control.cell_index%num_colors,)
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