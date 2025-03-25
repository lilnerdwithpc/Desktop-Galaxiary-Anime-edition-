import sys
import math
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QVector2D, QVector3D
from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF

idk = 0  # Global variable for animation

args = {
    'physics_fps': 60
}

class DesktopGalaxiary(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent background
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Click-through window
        self.showFullScreen()  # Full screen overlay

        self.setup_update_timers()

    def setup_update_timers(self):
        physics_fps = self.args['physics_fps']
        physics_dt = int(1000 / physics_fps)  # Convert FPS to milliseconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_physics_update)
        self.timer.start(physics_dt)

    def on_physics_update(self):
        global idk  
        idk += 1  # Move 5 pixels per frame
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():  # Ensure painter is active
            return
        
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        
        def _draw_joint(a: QPoint, b: QPoint, scale: int):
            '''
            Draws two circles at both ends, and a rectangle connecting them
            '''
            radius = scale // 2

            # Draw circles at both ends
            painter.drawEllipse(a.x() - radius, a.y() - radius, scale, scale)
            painter.drawEllipse(b.x() - radius, b.y() - radius, scale, scale)

            # Draw rotated rectangle
            painter.save()
            mid = (QPointF(a) + QPointF(b)) / 2  # Compute center point
            delta = b - a
            length = math.hypot(delta.x(), delta.y())
            angle = math.degrees(math.atan2(delta.y(), delta.x()))  # Correct angle calculation

            painter.translate(mid)  # Move origin to center
            painter.rotate(angle)  # Rotate to match line direction
            painter.drawRect(int(-length/2), int(-radius), int(length), int(scale))  # Centered rectangle
            painter.restore()

        global idk
        painter.setBrush(QColor(255, 221, 158, 255))
        _draw_joint(QPoint(100, 200), QPoint(300+idk, 300), 30)
   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesktopGalaxiary(args)
    sys.exit(app.exec_())
