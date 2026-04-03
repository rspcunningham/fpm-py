# qtlib.py
import sys
import math
import numpy as np
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPen, QBrush, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QFileDialog,
    QWidget, QGridLayout, QSizePolicy
)


class SquareRoiItem(QGraphicsRectItem):
    HANDLE_NONE = 0
    HANDLE_MOVE = 1
    HANDLE_TOPLEFT = 2
    HANDLE_TOPRIGHT = 3
    HANDLE_BOTTOMLEFT = 4
    HANDLE_BOTTOMRIGHT = 5

    def __init__(self, rect, bounds_rect, parent=None):
        super().__init__(rect, parent)
        self.bounds_rect = bounds_rect
        self.handle_size = 8.0
        self.min_size = 5.0

        self.setPen(QPen(Qt.red, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.setZValue(10)

        self.active_handle = self.HANDLE_NONE
        self.drag_start_pos = QPointF()
        self.drag_start_rect = QRectF()

        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

    def handle_rects(self):
        r = self.rect()
        hs = self.handle_size
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

        return {
            self.HANDLE_TOPLEFT: QRectF(x1 - hs/2, y1 - hs/2, hs, hs),
            self.HANDLE_TOPRIGHT: QRectF(x2 - hs/2, y1 - hs/2, hs, hs),
            self.HANDLE_BOTTOMLEFT: QRectF(x1 - hs/2, y2 - hs/2, hs, hs),
            self.HANDLE_BOTTOMRIGHT: QRectF(x2 - hs/2, y2 - hs/2, hs, hs),
        }

    def detect_handle(self, pos):
        for handle, rect in self.handle_rects().items():
            if rect.contains(pos):
                return handle
        if self.rect().contains(pos):
            return self.HANDLE_MOVE
        return self.HANDLE_NONE

    def hoverMoveEvent(self, event):
        handle = self.detect_handle(event.pos())

        cursor_map = {
            self.HANDLE_TOPLEFT: Qt.SizeFDiagCursor,
            self.HANDLE_BOTTOMRIGHT: Qt.SizeFDiagCursor,
            self.HANDLE_TOPRIGHT: Qt.SizeBDiagCursor,
            self.HANDLE_BOTTOMLEFT: Qt.SizeBDiagCursor,
            self.HANDLE_MOVE: Qt.SizeAllCursor,
            self.HANDLE_NONE: Qt.ArrowCursor,
        }
        self.setCursor(cursor_map[handle])
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.active_handle = self.detect_handle(event.pos())
            self.drag_start_pos = event.scenePos()
            self.drag_start_rect = QRectF(self.rect())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_handle == self.HANDLE_NONE:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self.drag_start_pos
        r0 = QRectF(self.drag_start_rect)

        if self.active_handle == self.HANDLE_MOVE:
            r = QRectF(r0)
            r.translate(delta)
            r = self.clamp_rect_to_bounds(r)
            self.setRect(r)
            event.accept()
            return

        r = self.square_from_drag(r0, delta, self.active_handle)
        r = self.clamp_square_to_bounds(r, self.active_handle)
        self.setRect(r.normalized())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.active_handle = self.HANDLE_NONE
        super().mouseReleaseEvent(event)

    def square_from_drag(self, r0, delta, handle):
        left = r0.left()
        right = r0.right()
        top = r0.top()
        bottom = r0.bottom()

        if handle == self.HANDLE_TOPLEFT:
            dx = delta.x()
            dy = delta.y()
            d = max(dx, dy)
            new_left = left + d
            new_top = top + d
            return QRectF(QPointF(new_left, new_top), QPointF(right, bottom)).normalized()

        if handle == self.HANDLE_TOPRIGHT:
            dx = delta.x()
            dy = delta.y()
            d = max(dx, -dy)
            new_right = right + d
            new_top = top - d
            return QRectF(QPointF(left, new_top), QPointF(new_right, bottom)).normalized()

        if handle == self.HANDLE_BOTTOMLEFT:
            dx = delta.x()
            dy = delta.y()
            d = max(-dx, dy)
            new_left = left - d
            new_bottom = bottom + d
            return QRectF(QPointF(new_left, top), QPointF(right, new_bottom)).normalized()

        if handle == self.HANDLE_BOTTOMRIGHT:
            dx = delta.x()
            dy = delta.y()
            d = max(dx, dy)
            new_right = right + d
            new_bottom = bottom + d
            return QRectF(QPointF(left, top), QPointF(new_right, new_bottom)).normalized()

        return r0

    def clamp_rect_to_bounds(self, r):
        b = self.bounds_rect

        if r.left() < b.left():
            r.moveLeft(b.left())
        if r.top() < b.top():
            r.moveTop(b.top())
        if r.right() > b.right():
            r.moveRight(b.right())
        if r.bottom() > b.bottom():
            r.moveBottom(b.bottom())

        return r

    def clamp_square_to_bounds(self, r, handle):
        b = self.bounds_rect
        size = max(self.min_size, min(r.width(), r.height()))

        if handle == self.HANDLE_TOPLEFT:
            anchor = self.drag_start_rect.bottomRight()
            max_size = min(anchor.x() - b.left(), anchor.y() - b.top())
            size = min(size, max_size)
            return QRectF(anchor.x() - size, anchor.y() - size, size, size)

        if handle == self.HANDLE_TOPRIGHT:
            anchor = self.drag_start_rect.bottomLeft()
            max_size = min(b.right() - anchor.x(), anchor.y() - b.top())
            size = min(size, max_size)
            return QRectF(anchor.x(), anchor.y() - size, size, size)

        if handle == self.HANDLE_BOTTOMLEFT:
            anchor = self.drag_start_rect.topRight()
            max_size = min(anchor.x() - b.left(), b.bottom() - anchor.y())
            size = min(size, max_size)
            return QRectF(anchor.x() - size, anchor.y(), size, size)

        if handle == self.HANDLE_BOTTOMRIGHT:
            anchor = self.drag_start_rect.topLeft()
            max_size = min(b.right() - anchor.x(), b.bottom() - anchor.y())
            size = min(size, max_size)
            return QRectF(anchor.x(), anchor.y(), size, size)

        return r

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        painter.setPen(QPen(Qt.red, 1))
        painter.setBrush(QBrush(Qt.white))
        for rect in self.handle_rects().values():
            painter.drawRect(rect)


class RoiGraphicsView(QGraphicsView):
    roiChanged = pyqtSignal(tuple)

    def __init__(self, image, parent=None):
        super().__init__(parent)

        self.image = np.asarray(image)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap = self.numpy_to_pixmap(self.image)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self.roi_item = None
        self.drawing = False
        self.draw_start = QPointF()

        self.image_rect = QRectF(self.pixmap.rect())
        self.scene.setSceneRect(self.image_rect)

        self.fitInView(self.image_rect, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.image_rect, Qt.KeepAspectRatio)

    def numpy_to_pixmap(self, arr):
        if arr.ndim == 2:
            h, w = arr.shape
            arr8 = arr.astype(np.float32)
            arr8 = 255 * (arr8 - arr8.min()) / (arr8.max() - arr8.min() + 1e-12)
            arr8 = arr8.astype(np.uint8)
            qimg = QImage(arr8.data, w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimg.copy())

        if arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            arr8 = arr.astype(np.uint8)
            qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())

        raise ValueError("Image must be HxW grayscale or HxWx3 RGB")

    def make_square_rect(self, p1, p2):
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        side = min(abs(dx), abs(dy))

        if dx >= 0:
            x2 = p1.x() + side
        else:
            x2 = p1.x() - side

        if dy >= 0:
            y2 = p1.y() + side
        else:
            y2 = p1.y() - side

        return QRectF(p1, QPointF(x2, y2)).normalized()

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())

            if self.roi_item is not None and item is self.roi_item:
                super().mousePressEvent(event)
                return

            if self.roi_item is not None:
                handle = self.roi_item.detect_handle(self.roi_item.mapFromScene(scene_pos))
                if handle != SquareRoiItem.HANDLE_NONE:
                    super().mousePressEvent(event)
                    return

            if self.image_rect.contains(scene_pos):
                self.drawing = True
                self.draw_start = scene_pos

                if self.roi_item is not None:
                    self.scene.removeItem(self.roi_item)
                    self.roi_item = None

                rect = QRectF(self.draw_start, self.draw_start).normalized()
                self.roi_item = SquareRoiItem(rect, self.image_rect)
                self.scene.addItem(self.roi_item)
                self.emit_roi()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.roi_item is not None:
            scene_pos = self.mapToScene(event.pos())
            x = min(max(scene_pos.x(), self.image_rect.left()), self.image_rect.right())
            y = min(max(scene_pos.y(), self.image_rect.top()), self.image_rect.bottom())
            rect = self.make_square_rect(self.draw_start, QPointF(x, y))
            self.roi_item.setRect(rect)
            self.emit_roi()
            return

        super().mouseMoveEvent(event)
        self.emit_roi()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.emit_roi()
            return

        super().mouseReleaseEvent(event)
        self.emit_roi()

    def emit_roi(self):
        roi = self.get_roi()
        if roi is not None:
            self.roiChanged.emit(roi)

    def get_roi(self):
        if self.roi_item is None:
            return None

        r = self.roi_item.rect().normalized()

        x = int(round(r.left()))
        y = int(round(r.top()))
        w = int(round(r.width()))
        h = int(round(r.height()))

        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))
        w = max(1, min(w, self.image.shape[1] - x))
        h = max(1, min(h, self.image.shape[0] - y))

        side = min(w, h)
        return (x, y, side, side)


class RoiDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Square ROI")
        self.resize(1000, 800)

        self.roi = None

        self.info_label = QLabel(
            "Drag to create square ROI. Drag corners to resize square. Drag inside to move."
        )

        self.view = RoiGraphicsView(image)
        self.view.roiChanged.connect(self.on_roi_changed)

        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.setEnabled(False)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def on_roi_changed(self, roi):
        self.roi = roi
        x, y, w, h = roi
        self.info_label.setText(
            f"ROI: x={x}, y={y}, w={w}, h={h}    (square locked)"
        )
        self.ok_button.setEnabled(True)


def select_roi(image):
    app = QApplication.instance()
    created_app = False

    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    dialog = RoiDialog(image)
    result = dialog.exec_()
    roi = dialog.roi

    if created_app:
        app.quit()

    if result == QDialog.Accepted and roi is not None:
        return roi

    return None






def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert a 2D image to uint8 for display.
    """
    if img.ndim != 2:
        raise ValueError("Image must be a 2D grayscale image")

    img = np.asarray(img, dtype=np.float64)

    vmin = np.min(img)
    vmax = np.max(img)

    if vmax <= vmin:
        return np.zeros(img.shape, dtype=np.uint8)

    out = 255.0 * (img - vmin) / (vmax - vmin)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_window_level(img: np.ndarray, window: float, level: float) -> np.ndarray:
    """
    Apply window/level to a grayscale image and return uint8 display image.

    Pixels mapped as:
        low  = level - window/2
        high = level + window/2
    Values <= low map to 0
    Values >= high map to 255
    """
    img = np.asarray(img, dtype=np.float64)

    if window <= 1e-12:
        window = 1e-12

    low = level - window / 2.0
    high = level + window / 2.0

    out = (img - low) * (255.0 / (high - low))
    return np.clip(out, 0, 255).astype(np.uint8)


def numpy_to_qimage_gray(img8: np.ndarray) -> QImage:
    """
    Convert a 2D uint8 grayscale NumPy image to QImage.
    """
    if img8.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if img8.ndim != 2:
        raise ValueError("Image must be 2D")

    h, w = img8.shape
    bytes_per_line = w
    return QImage(
        img8.data, w, h, bytes_per_line, QImage.Format_Grayscale8
    ).copy()


class ImageView(QGraphicsView):
    def __init__(self, image_array: np.ndarray, info_label: QLabel, title="", parent=None):
        super().__init__(parent)

        self.image_array = np.asarray(image_array, dtype=np.float64)
        if self.image_array.ndim != 2:
            raise ValueError("Each image must be a 2D grayscale array")

        self.info_label = info_label
        self.title = title

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.display_image = None
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.zoom_factor = 1.25
        self.min_scale = 0.05
        self.max_scale = 50.0
        self.current_scale = 1.0

        # Window/level state
        self.img_min = float(np.min(self.image_array))
        self.img_max = float(np.max(self.image_array))
        self.img_range = max(self.img_max - self.img_min, 1.0)

        self.window = self.img_range
        self.level = 0.5 * (self.img_min + self.img_max)

        self.right_dragging = False
        self.right_drag_start = QPoint()
        self.window_start = self.window
        self.level_start = self.level

        self.update_display()

    def update_display(self):
        self.display_image = apply_window_level(
            self.image_array, self.window, self.level
        )
        qimg = numpy_to_qimage_gray(self.display_image)
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def reset_window_level(self):
        self.window = self.img_range
        self.level = 0.5 * (self.img_min + self.img_max)
        self.update_display()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = self.zoom_factor
        else:
            factor = 1.0 / self.zoom_factor

        new_scale = self.current_scale * factor
        if self.min_scale <= new_scale <= self.max_scale:
            self.scale(factor, factor)
            self.current_scale = new_scale

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_dragging = True
            self.right_drag_start = event.pos()
            self.window_start = self.window
            self.level_start = self.level
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_dragging = False
            self.unsetCursor()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.RightButton:
            self.reset_window_level()
            self.update_info_label(event.pos())
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.right_dragging:
            delta = event.pos() - self.right_drag_start

            # Horizontal motion adjusts window
            # Vertical motion adjusts level
            window_sensitivity = self.img_range / 300.0
            level_sensitivity = self.img_range / 300.0

            new_window = self.window_start + delta.x() * window_sensitivity
            new_level = self.level_start - delta.y() * level_sensitivity

            self.window = max(new_window, 1e-6)
            self.level = new_level

            self.update_display()
            self.update_info_label(event.pos(), show_wl=True)
            event.accept()
            return

        self.update_info_label(event.pos(), show_wl=False)
        super().mouseMoveEvent(event)

    def update_info_label(self, view_pos, show_wl=False):
        scene_pos = self.mapToScene(view_pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        h, w = self.image_array.shape

        prefix = f"{self.title}: " if self.title else ""

        if 0 <= x < w and 0 <= y < h:
            pixel_value = self.image_array[y, x]

            if float(pixel_value).is_integer():
                pixel_text = str(int(pixel_value))
            else:
                pixel_text = f"{pixel_value:.3f}"

            text = f"{prefix}{x}, {y}, {pixel_text}"

            if show_wl:
                text += f", W={self.window:.3f}, L={self.level:.3f}"

            self.info_label.setText(text)
        else:
            text = ""#f"{prefix}Outside image"
            if show_wl:
                text += f", W={self.window:.3f}, L={self.level:.3f}"
            self.info_label.setText(text)

    def fit_initial_view(self):
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.current_scale = 1.0


class ImagePanel(QWidget):
    def __init__(self, image_array: np.ndarray, title="", parent=None):
        super().__init__(parent)

#        self.title_label = QLabel(title)
#        self.info_label = QLabel("Move mouse over image")
#        self.view = ImageView(image_array, self.info_label, title=title)

#        layout = QVBoxLayout(self)
#        layout.addWidget(self.title_label)
#        layout.addWidget(self.view)
#        layout.addWidget(self.info_label)

        #self.info_label = QLabel("Move mouse over image")
        self.info_label = QLabel("")
        self.view = ImageView(image_array, self.info_label, title=title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        if title:
            self.title_label = QLabel(title)
            layout.addWidget(self.title_label)

        layout.addWidget(self.view)
        layout.addWidget(self.info_label)


    def fit_initial_view(self):
        self.view.fit_initial_view()


class MultiImageWindow(QMainWindow):
    def __init__(self, images, nrows=None, ncols=None, titles=None):
        super().__init__()
        #self.setWindowTitle("Multiple Grayscale Images")

        n_images = len(images)
        if n_images == 0:
            raise ValueError("images list is empty")

        if titles is None:
            #titles = [f"Image {i}" for i in range(n_images)]
            titles = [""] * n_images

        if len(titles) != n_images:
            raise ValueError("titles must have same length as images")

        if nrows is None and ncols is None:
            ncols = math.ceil(math.sqrt(n_images))
            nrows = math.ceil(n_images / ncols)
        elif nrows is None:
            nrows = math.ceil(n_images / ncols)
        elif ncols is None:
            ncols = math.ceil(n_images / nrows)

        central = QWidget()
        self.setCentralWidget(central)

        self.grid = QGridLayout(central)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setHorizontalSpacing(2)
        self.grid.setVerticalSpacing(2)
        self.panels = []

        for i, img in enumerate(images):
            img = np.asarray(img)
            if img.ndim != 2:
                raise ValueError(f"Image {i} must be 2D grayscale")

            row = i // ncols
            col = i % ncols

            panel = ImagePanel(img, title=titles[i])
            self.grid.addWidget(panel, row, col)
            self.panels.append(panel)

        self.resize(2400, 1400)

    def showEvent(self, event):
        super().showEvent(event)
        for panel in self.panels:
            panel.fit_initial_view()


def show_grayscale_subplots(images, nrows=None, ncols=None, titles=None):
    """
    Display several grayscale images in a PyQt5 grid, similar to plt.subplots().

    Controls for each image:
      - mouse wheel: zoom
      - left-drag: pan
      - right-drag: change window/level
      - right double-click: reset window/level
    """
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    win = MultiImageWindow(images, nrows=nrows, ncols=ncols, titles=titles)
    win.show()

    if owns_app:
        sys.exit(app.exec_())

    return win
