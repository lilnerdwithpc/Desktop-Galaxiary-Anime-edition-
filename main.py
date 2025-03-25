import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt

class DesktopGalaxiary(QWidget):
    def __init__(self):
        super().__init__()

        # Make the window full screen
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent background
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Click-through window
        self.showFullScreen()  # Make it full screen

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a red square in the center
        square_size = 100  # Size of the square
        x = (self.width() - square_size) // 2
        y = (self.height() - square_size) // 2
        painter.setBrush(QColor(255, 0, 0, 255))  # Red color
        painter.drawRect(x, y, square_size, square_size)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = DesktopGalaxiary()
    sys.exit(app.exec_())
