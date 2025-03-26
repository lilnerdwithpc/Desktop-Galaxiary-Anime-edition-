import sys
import math
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QVector2D, QVector3D, QQuaternion
from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF

idk = 0  # Global variable for animation

args = {
    'physics_fps': 60
}

class RigElement:
    def __init__(self, name:str, c0:QVector3D=None, c1:QVector3D=None, rotation:QQuaternion=None):
        self.name = name
        self.c0 = c0 if c0 is not None else QVector3D()
        self.c1 = c1 if c1 is not None else QVector3D()
        self.rotation = rotation if rotation is not None else QQuaternion()

        self._object_position = QVector3D()
        self._object_rotation = QQuaternion()
        self._world_position = QVector3D()
        self.parent:RigElement = None
        self.childs:list[RigElement] = []

    def add_child(self, child:'RigElement'):
        # TODO: add_child should only be performed once for each child. else it malfunctions, and i wont fix it. why? because yes. why todo? because i might fix it sometime.
        self.childs.append(child)
        child.parent = self

    def add_childs(self, childs:list['RigElement']):
        # TODO: add_childs should only be performed once for each child. else it malfunctions, and i wont fix it. why? because yes. why todo? because i might fix it sometime.
        for child in childs:
            self.childs.append(child)
            child.parent = self

class Rig:
    def __init__(self, position:QVector3D, scale:float):
        '''
        - Root
            - UpperTorso
                - Head
                - LeftUpperArm
                    - LeftLowerArm
                        - LeftHand
                - RightUpperArm
                    - RightLowerArm
                        - RightHand
                - LowerTorso
                    - LeftUpperLeg
                        - LeftLowerLeg
                            - LeftFoot
                    - RightUpperLeg
                        - RightLowerLeg
                            - RightFoot
        '''
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

    def update_position(self):
        def _recurve(rig_element:RigElement):
            for child in rig_element.childs:
                child._object_rotation = child.rotation * rig_element._object_rotation
                child._object_position = rig_element._object_position + rig_element._object_rotation.rotatedVector(child.c0)*self.scale
                
                child._world_position = self.Root._world_position + self.Root.rotation.rotatedVector(child._object_position) + child.rotation.rotatedVector(child.c1)*self.scale
                _recurve(child)
        _recurve(self.Root)

    def get_layout(self):
        layout:dict[RigElement,dict] = {self.Root : {}}
        def _recurve(rig_element:RigElement, childs_dict:dict):
            for child in rig_element.childs:
                childs_dict[child] = {}
                _recurve(child, childs_dict[child])
        _recurve(self.Root, layout[self.Root])
        return layout

    def get_layout_string(self):
        layout = self.get_layout()
        layout_str = ''
        def _recurve(layout_sect:dict[RigElement,dict], depth:int):
            nonlocal layout_str
            for k, v in layout_sect.items():
                layout_str += '\033[38;5;235m|\033[0m   '*depth + k.name + '\n'
                _recurve(v, depth+1)
        _recurve(layout, 0)
        return layout_str

class DesktopGalaxiary(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.rig = Rig(QVector3D(100, 100, 0), 0.3)

        #self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        #self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent background
        #self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Click-through window
        #self.showFullScreen()  # Full screen overlay

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

        def _draw_rig(rig:Rig):
            rig.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), 1)
            rig.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 1.1)
            rig.LeftUpperArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 1.3)
            rig.LeftLowerArm.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 1)
            rig.LeftUpperLeg.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 1)
            rig.LeftLowerLeg.rotation *= QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 1)
            def _recurve(rig_element:RigElement):
                """ Recursively draw all rig elements """
                for child in rig_element.childs:
                    start = QPoint(int(child._world_position.x()), int(child._world_position.y()))
                    end = QPoint(int(rig_element._world_position.x()), int(rig_element._world_position.y()))
                    _draw_joint(start, end, int(rig.scale*100))  # Adjust scale as needed
                    _recurve(child)  # Recurse
            root = rig.Root
            _recurve(root)

        # Update rig world positions before drawing
        self.rig.update_position()

        # Set color for joints
        painter.setBrush(QColor(255, 221, 158, 255))

        # Draw the rig starting from the root
        _draw_rig(self.rig)

        painter.end()
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesktopGalaxiary(args)
    sys.exit(app.exec_())
