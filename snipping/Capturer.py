from PySide6.QtWidgets import QWidget, QApplication, QRubberBand
from PySide6.QtGui import QMouseEvent, QGuiApplication
from PySide6.QtCore import Qt, QPoint, QRect
import time

class Capture(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.main.hide()
        
        self.setMouseTracking(True)
        # Use primaryScreen instead of desktop
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.setGeometry(0, 0, screen_geometry.width(), screen_geometry.height())
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowOpacity(0.15)

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()

        QApplication.setOverrideCursor(Qt.CrossCursor)
        
        time.sleep(0.31)
        # 0 means entire screen in grabWindow
        self.imgmap = screen.grabWindow(0)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
            self.rubber_band.show() 

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event.button() == Qt.LeftButton:
            self.rubber_band.hide()
            
            rect = self.rubber_band.geometry()
            self.imgmap = self.imgmap.copy(rect)
            QApplication.restoreOverrideCursor()

            # set clipboard
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(self.imgmap)

            self.imgmap.save("TEST.png")

            self.main.label.setPixmap(self.imgmap)
            self.main.show()

            self.close()