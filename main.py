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

        self._world_position = QVector3D()
        self.parent = None
        self.childs = []

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
    def __init__(self, pos:QVector2D):
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

    def update_world_position(self):
        pass

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

        self.rig = Rig(QVector2D())

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

        global idk
        painter.setBrush(QColor(255, 221, 158, 255))
        _draw_joint(QPoint(100, 200), QPoint(300+idk, 300), 30)
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesktopGalaxiary(args)
    sys.exit(app.exec_())
