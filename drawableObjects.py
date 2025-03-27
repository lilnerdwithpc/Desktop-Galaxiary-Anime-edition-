import math
import random
import copy
from PyQt5.QtGui import QPainter, QColor, QVector2D, QVector3D, QQuaternion
from PyQt5.QtCore import QPoint, QPointF, QRectF

from typing import Protocol

class ObjectLike(Protocol):
    delta_position: QVector3D
    delta_rotation:QQuaternion
    scale:QVector3D
    color:QColor

    def draw(self, painter: QPainter, position: QVector3D, rotation: QQuaternion) -> None:
        pass

class Cylinder:
    def __init__(self, delta_position:QVector3D, delta_rotation:QQuaternion, scale:QVector3D, color:QColor):
        self.delta_position = delta_position
        self.delta_rotation = delta_rotation
        self.scale = scale
        self.color = color
        
    def draw(self, painter: QPainter, position: QVector3D, rotation: QQuaternion):
        # Calculate global position and rotation
        rot = rotation * self.delta_rotation
        pos = position + rot.rotatedVector(self.delta_position)

        # Compute the rotated X-axis – this gives the orientation of the object in the XY plane.
        rotated_x = rot.rotatedVector(QVector3D(0, 1, 0))
        angle_cam = math.degrees(math.atan2(rotated_x.y(), rotated_x.x()))
        rotated_x_projected_unit = min(math.hypot(rotated_x.x(), rotated_x.y()), 1) # To prevent float's dark magic making it larger than 1.

        painter.save()
        painter.setBrush(self.color)

        # Translate to the cylinder's position (using x, y from the QVector3D)
        painter.translate(pos.x(), pos.y())
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

class Capsule:
    def __init__(self, delta_position:QVector3D, delta_rotation:QQuaternion, scale:QVector3D, color:QColor):
        self.delta_position = delta_position
        self.delta_rotation = delta_rotation
        self.scale = scale
        self.color = color
        
    def draw(self, painter: QPainter, position: QVector3D, rotation: QQuaternion):
        # Calculate global position and rotation
        rot = rotation * self.delta_rotation
        pos = position + rot.rotatedVector(self.delta_position)

        # Compute the rotated X-axis – this gives the orientation of the object in the XY plane.
        rotated_x = rot.rotatedVector(QVector3D(0, 1, 0))
        angle_cam = math.degrees(math.atan2(rotated_x.y(), rotated_x.x()))
        rotated_x_projected_unit = min(math.hypot(rotated_x.x(), rotated_x.y()), 1) # To prevent float's dark magic making it larger than 1.

        painter.save()
        painter.setBrush(self.color)

        # Translate to the cylinder's position (using x, y from the QVector3D)
        painter.translate(pos.x(), pos.y())
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

class Sphere:
    def __init__(self, delta_position:QVector3D, delta_rotation:QQuaternion, scale:QVector3D, color:QColor):
        self.delta_position = delta_position
        self.delta_rotation = delta_rotation
        self.scale = scale
        self.color = color
        
    def draw(self, painter: QPainter, position: QVector3D, rotation: QQuaternion):
        # Calculate global position and rotation
        rot = rotation * self.delta_rotation
        pos = position + rot.rotatedVector(self.delta_position)

        radius = min(self.scale.x(), self.scale.y(), self.scale.z()) / 2

        painter.save()
        painter.setBrush(self.color)
        painter.drawEllipse(QPointF(pos.x(), pos.y()), radius, radius)
        painter.restore()