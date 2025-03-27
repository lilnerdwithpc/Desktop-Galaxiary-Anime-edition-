import sys
import math
import random
import copy
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QVector2D, QVector3D, QQuaternion
from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF, QRectF
import drawableObjects as dObj


class Drawable:
    '''
    Container for group of drawableObjects to be drawn
    '''
    def __init__(self, objects:list[dObj.ObjectLike]):
        self.objects = objects

    def update(self, position:QVector3D, rotation:QQuaternion):
        for object in self.objects:
            object.update(position, rotation)

class RigElement:
    def __init__(self, name:str, c0:QVector3D=None, c1:QVector3D=None, rotation:QQuaternion=None, drawable:Drawable=None):
        self.name = name
        self.c0 = c0 if c0 is not None else QVector3D()
        self.c1 = c1 if c1 is not None else QVector3D()
        self.rotation = rotation if rotation is not None else QQuaternion()

        self.drawable = drawable

        self._object_position = QVector3D()
        self._object_rotation = QQuaternion()
        self._world_position = QVector3D()
        self._world_rotation = QQuaternion()
        self.parent:RigElement = None
        self.childs:list[RigElement] = []

    def add_child(self, child:'RigElement'):
        # TODO: add_child should only be performed once for each child. else it malfunctions, and i wont fix it. why? because yes.
        self.childs.append(child)
        child.parent = self

    def add_childs(self, childs:list['RigElement']):
        # TODO: add_childs should only be performed once for each child. else it malfunctions, and i wont fix it. why? because yes.
        for child in childs:
            self.childs.append(child)
            child.parent = self

    def updateDrawables(self):
        '''
        Update Drawables using current metadata
        '''
        if self.drawable:
            self.drawable.update(self._world_position, self._world_rotation)

class Rig:
    def __init__(self, position:QVector3D, scale:float):
        # Create the erm rig elements
        self.Root = RigElement('Root')

        self.UpperTorso = RigElement('UpperTorso')
        self.LowerTorso = RigElement('LowerTorso', c0=QVector3D(0, 168, 0))

        self.Head = RigElement('Head', c0=QVector3D(0, -58, 0))

        self.LeftUpperArm = RigElement('LeftUpperArm', c0=QVector3D(110, 0, 0))
        self.LeftLowerArm = RigElement('LeftLowerArm', c0=QVector3D(0, 140, 0))
        self.LeftHand = RigElement('LeftHand', c0=QVector3D(0, 140, 0))

        self.RightUpperArm = RigElement('RightUpperArm', c0=QVector3D(-110, 0, 0))
        self.RightLowerArm = RigElement('RightLowerArm', c0=QVector3D(0, 140, 0))
        self.RightHand = RigElement('RightHand', c0=QVector3D(0, 140, 0))

        self.LeftUpperLeg = RigElement('LeftUpperLeg', c0=QVector3D(55, 169, 0))
        self.LeftLowerLeg = RigElement('LeftLowerLeg', c0=QVector3D(0, 143, 0))
        self.LeftFoot = RigElement('LeftFoot', c0=QVector3D(0, 143, 0))

        self.RightUpperLeg = RigElement('RightUpperLeg', c0=QVector3D(-55, 169, 0))
        self.RightLowerLeg = RigElement('RightLowerLeg', c0=QVector3D(0, 143, 0))
        self.RightFoot = RigElement('RightFoot', c0=QVector3D(0, 143, 0))


        self.Root._world_position = position
        self.position = self.Root._world_position
        self.rotation = self.Root.rotation
        self.scale = scale


        # Connect the rig elements
        self.Root.add_child(self.UpperTorso)
        
        self.UpperTorso.add_childs((self.Head, self.LeftUpperArm, self.RightUpperArm, self.LowerTorso))
        self.LowerTorso.add_childs((self.LeftUpperLeg, self.RightUpperLeg))

        self.LeftUpperArm.add_child(self.LeftLowerArm)
        self.RightUpperArm.add_child(self.RightLowerArm)

        self.LeftLowerArm.add_child(self.LeftHand)
        self.RightLowerArm.add_child(self.RightHand)

        self.RightUpperLeg.add_child(self.RightLowerLeg)
        self.LeftUpperLeg.add_child(self.LeftLowerLeg)
        
        self.RightLowerLeg.add_child(self.RightFoot)
        self.LeftLowerLeg.add_child(self.LeftFoot)


        # Setup Drawables
        _color_pallete = {
            'skin': QColor(255, 231, 220, 255),
            'shirt': QColor(68, 64, 63, 255),
            'pant': QColor(61, 59, 62, 255),
            'lavender': QColor(99, 61, 144, 255),
            'hair': QColor(66, 30, 99, 255)
        }

        # LeftArm
        #region
        self.LeftUpperArm.drawable = Drawable([
            dObj.Capsule(
                delta_position=QVector3D(0, 70, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(80, 140, 80), 
                color=_color_pallete['shirt']
            ),
        ])
        self.LeftLowerArm.drawable = Drawable([
            dObj.Cylinder(
                delta_position=QVector3D(0, 70, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(80, 140, 80), 
                color=_color_pallete['shirt']
            ),
            dObj.Cylinder(
                delta_position=QVector3D(0, 110, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(94, 60, 94), 
                color=_color_pallete['shirt']
            )
        ])
        self.LeftHand.drawable = Drawable([
            dObj.Sphere(
                delta_position=QVector3D(0, 20, 0),
                delta_rotation=QQuaternion(),
                scale=QVector3D(90, 90, 90), 
                color=_color_pallete['skin']
            )
        ])
        #endregion

        # RightArm
        #region
        self.RightUpperArm.drawable = Drawable([
            dObj.Capsule(
                delta_position=QVector3D(0, 70, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(80, 140, 80), 
                color=_color_pallete['shirt']
            ),
        ])
        self.RightLowerArm.drawable = Drawable([
            dObj.Cylinder(
                delta_position=QVector3D(0, 70, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(80, 140, 80), 
                color=_color_pallete['shirt']
            ),
            dObj.Cylinder(
                delta_position=QVector3D(0, 110, 0), 
                delta_rotation=QQuaternion(),
                scale=QVector3D(94, 60, 94), 
                color=_color_pallete['shirt']
            )
        ])
        self.RightHand.drawable = Drawable([
            dObj.Sphere(
                delta_position=QVector3D(0, 20, 0),
                delta_rotation=QQuaternion(),
                scale=QVector3D(90, 90, 90), 
                color=_color_pallete['skin']
            )
        ])
        #endregion


        for element in self.get_elements():
            if element.drawable:
                for drawable in element.drawable.objects:
                    drawable.delta_position *= self.scale
                    drawable.scale *= self.scale

    def update_position(self):
        '''
        Update the position of each RigElement in Rig.
        '''
        def _recurse(rig_element:RigElement):
            for child in rig_element.childs:
                child._object_rotation = rig_element._object_rotation * child.rotation
                child._object_position = rig_element._object_position + rig_element._object_rotation.rotatedVector(child.c0)*self.scale
                
                child._world_rotation = self.Root.rotation * child._object_rotation
                child._world_position = self.Root._world_position + self.Root.rotation.rotatedVector(child._object_position) + child._world_rotation.rotatedVector(child.c1)*self.scale
                _recurse(child)
        _recurse(self.Root)

    def updateDrawables(self):
        '''
        Update Drawables in each RigElement using current metadata
        '''
        elements = self.get_elements()
        for rig_element in elements:
            rig_element.updateDrawables()

    def draw(self, painter:QPainter):
        '''
        Draw the Rig.
        '''
        # Update Drawables before drawing
        self.updateDrawables()

        # Get and sort dobjs by what should be in front of what
        dobjs = self.get_drawable_objects()
        dobjs.sort(key=lambda v: v.position.z(), reverse=True)
        for o in dobjs:
            o.draw(painter)

    def get_drawable_objects(self) -> list[dObj.ObjectLike]:
        dobjs = []
        elements = self.get_elements()
        for rig_element in elements:
            if rig_element.drawable:
                dobjs.extend(rig_element.drawable.objects)
        return dobjs

    def get_elements(self) -> list[RigElement]:
        '''
        Returns a iterable over all element in Rig
        '''
        elements = []
        def _recurse(rig_element:RigElement):
            for child in rig_element.childs:
                elements.append(child)
                _recurse(child)
        _recurse(self.Root)
        return elements

    def get_layout_dict(self):
        '''
        Returns a layout dict of Rig
        '''
        layout:dict[RigElement,dict] = {self.Root : {}}
        def _recurse(rig_element:RigElement, childs_dict:dict):
            for child in rig_element.childs:
                childs_dict[child] = {}
                _recurse(child, childs_dict[child])
        _recurse(self.Root, layout[self.Root])
        return layout

    def get_layout_string(self):
        '''
        Returns a string representing layout dict of Rig
        '''
        layout = self.get_layout_dict()
        layout_str = ''
        def _recurse(layout_sect:dict[RigElement,dict], depth:int):
            nonlocal layout_str
            for k, v in layout_sect.items():
                layout_str += '\033[38;5;235m|\033[0m   '*depth + k.name + '\n'
                _recurse(v, depth+1)
        _recurse(layout, 0)
        return layout_str
    
    def draw_bones(self, painter:QPainter):
        '''
        Draw the rig's bones ig
        '''
        def _draw_line(a:QPoint, b:QPoint, scale:int):
            '''
            Draws a line connecting a and b
            '''
            radius = scale // 2

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
        
        elements = self.get_elements()
        elements.sort(key=lambda v: v._world_position.z(), reverse=True)
        ae = 40
        for rig_element in elements:
            ae += 14
            painter.setBrush(QColor(255-ae, ae, 0, 255)) 
            start = QPointF(rig_element._world_position.x(), rig_element._world_position.y())
            end = rig_element._world_rotation.rotatedVector(QVector3D(0, 7, 0))
            end = start + QPointF(end.x(), end.y())
            painter.drawEllipse(start, 2, 2)
            _draw_line(start, end, 2)  # Adjust scale as needed

class DesktopGalaxiary(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.rig = Rig(QVector3D(300, 200, 0), args['scale'])

        #self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        #self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent background
        #self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Click-through window
        #self.showFullScreen()  # Full screen overlay

        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")
        self.show()

        self.setup_update_timers()

    def setup_update_timers(self):
        physics_fps = self.args['physics_fps']
        physics_dt = int(1000 / physics_fps)  # Convert FPS to milliseconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_physics_update)
        self.timer.start(physics_dt)

    def on_physics_update(self):
        self.rig.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), 1)
        self.rig.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 0.6)
        self.rig.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 0.16)
        self.rig.LeftUpperArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 3)
        self.rig.LeftLowerArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 1)
        self.rig.RightUpperArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 1.7)
        self.rig.RightLowerArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 0.8)
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # Update rig world positions before drawing
        self.rig.update_position()
        
        self.rig.draw(painter)
        self.rig.draw_bones(painter)

        painter.end()
    


if __name__ == "__main__":
    args = {
        'physics_fps': 60,
        'scale': 0.3
    }
    
    app = QApplication(sys.argv)
    window = DesktopGalaxiary(args)
    sys.exit(app.exec_())
