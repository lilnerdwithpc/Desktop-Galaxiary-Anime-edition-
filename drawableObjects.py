import math
import random
import copy
from PyQt5.QtGui import QPainter, QColor, QVector2D, QVector3D, QQuaternion
from PyQt5.QtCore import QPoint, QPointF, QRectF

class ObjectLike:
    def __init__(self, delta_position:QVector3D, delta_rotation:QQuaternion, scale:QVector3D, color:QColor) -> None:
        self.delta_position = delta_position
        self.delta_rotation = delta_rotation
        self.scale = scale
        self.color = color
        self.position = None
        self.rotation = None

    def update(self, position: QVector3D, rotation: QQuaternion) -> None:
        # Calculate global position and rotation
        self.rotation = rotation * self.delta_rotation
        self.position = position + self.rotation.rotatedVector(self.delta_position)
        
    def draw(self, painter: QPainter) -> None:
        pass

class Cylinder(ObjectLike):
    def draw(self, painter: QPainter):
        '''
        Draws the object using current position and rotation.
        Make sure to update position and rotation with updatePosition()
        '''
        # Compute the rotated X-axis – this gives the orientation of the object in the XY plane.
        rotated_x = self.rotation.rotatedVector(QVector3D(0, 1, 0))
        angle_cam = math.degrees(math.atan2(rotated_x.y(), rotated_x.x()))
        rotated_x_projected_unit = min(math.hypot(rotated_x.x(), rotated_x.y()), 1) # To prevent float's dark magic making it larger than 1.

        painter.save()
        painter.setBrush(self.color)

        # Translate to the cylinder's position (using x, y from the QVector3D)
        painter.translate(self.position.x(), self.position.y())
        painter.rotate(angle_cam)
        
        # Define dimensions based on scale:
        diameter = int(self.scale.x())
        length = int(self.scale.y())

        # Ohio pre-calculation of skibidi.
        morphed_length_rect = length * rotated_x_projected_unit
        morphed_length_elipse = diameter * (1 - rotated_x_projected_unit**2)**0.5
        morphed_length_rect_over_2 = morphed_length_rect / 2
        morphed_length_elipse_over_2 = morphed_length_elipse / 2

        morphed_length_rect = int(morphed_length_rect)
        morphed_length_rect_over_2 = int(morphed_length_rect_over_2)

        # Draw the shapes (a central rectangle and ellipses at the ends)
        #painter.drawRect(-400, -1, 800, 2)
        painter.drawRect(-morphed_length_rect_over_2, -diameter // 2, morphed_length_rect, diameter)
        painter.drawEllipse(QPointF(morphed_length_rect_over_2, 0), morphed_length_elipse_over_2, diameter / 2)
        painter.drawEllipse(QPointF(-morphed_length_rect_over_2, 0), morphed_length_elipse_over_2, diameter / 2)
        
        painter.restore()

class Capsule(ObjectLike):
    def draw(self, painter: QPainter):
        '''
        Draws the object using current position and rotation.
        Make sure to update position and rotation with updatePosition()
        '''
        # Compute the rotated X-axis – this gives the orientation of the object in the XY plane.
        rotated_x = self.rotation.rotatedVector(QVector3D(0, 1, 0))
        angle_cam = math.degrees(math.atan2(rotated_x.y(), rotated_x.x()))
        rotated_x_projected_unit = min(math.hypot(rotated_x.x(), rotated_x.y()), 1) # To prevent float's dark magic making it larger than 1.

        painter.save()
        painter.setBrush(self.color)

        # Translate to the cylinder's position (using x, y from the QVector3D)
        painter.translate(self.position.x(), self.position.y())
        painter.rotate(angle_cam)
        
        # Define dimensions based on scale:
        diameter = int(self.scale.x())
        length = int(self.scale.y())

        # Ohio pre-calculation of skibidi.
        morphed_length_rect = length * rotated_x_projected_unit
        morphed_length_rect_over_2 = morphed_length_rect / 2

        morphed_length_rect = int(morphed_length_rect)
        morphed_length_rect_over_2 = int(morphed_length_rect_over_2)

        # Draw the shapes (a central rectangle and ellipses at the ends)
        #painter.drawRect(-400, -1, 800, 2)
        painter.drawRect(-morphed_length_rect_over_2, -diameter // 2, morphed_length_rect, diameter)
        painter.drawEllipse(QPointF(morphed_length_rect_over_2, 0), diameter/2, diameter / 2)
        painter.drawEllipse(QPointF(-morphed_length_rect_over_2, 0), diameter/2, diameter / 2)
        
        painter.restore()

class Sphere(ObjectLike):
    def draw(self, painter: QPainter):
        '''
        Draws the object using current position and rotation.
        Make sure to update position and rotation with updatePosition()
        '''
        radius = min(self.scale.x(), self.scale.y(), self.scale.z()) / 2

        painter.save()
        painter.setBrush(self.color)
        painter.drawEllipse(QPointF(self.position.x(), self.position.y()), radius, radius)
        painter.restore()