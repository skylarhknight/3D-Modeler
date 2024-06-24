"""
Viewer Class for 3D Modeller

"""

# Import Packages
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Viewer(object):
    
    # Initialize the viewer
    def __init__(self, model):
        self.init_interface()
        self.init_opengl()
        self.init_scene()
        self.init_interaction()
        init_primitives()
        
    # Initizalize the window + register the render function
    def init_interface(self):
        glutInit()
        glutInitWindowSize(640, 480)
        gluCreateWindow("3D Modeller")
        gluInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        gluDisplayFunc(self.render)
        
    # Set up OpenGL State
    def init_opengl(self):
        self.inverseModelView = np.identity(4)
        self.modelView = np.identity(4)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BACK)
        glEnble(GL_LESS)
        
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT, GL_POSITION, GLfloat_4(0, 0, 1, 0))
        glLightfv(GL_LIGHT, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.4, 0.4, 0.4, 0.0)
        
    def init_scene(self):
        self.scene = Scene()
        self.create_sample_scene()
        
    def create_sample_scene(self):
        
        # Cube
        cube_node = Cube()
        cube_node.translate(2, 0 , 2)
        cube_node.color_index = 2
        self.scene.add_node(cube_node)
        
        # Sphere
        sphere_node = Sphere()
        sphere_node.translate(-2, 0, 2)
        sphere_node.color_index = 3
        self.scene.add_node(sphere_node)
        
        hierarchical_node = SnowFiigure()
        hierarchical_node.translate(-2, 0, -2)
        self.scene.add_node(hierarchical_node)
    
    # Interaction : Init user interaction and callbacks
    def init_interaction(self):
        self.interaction = Interaction()
        self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)  
        self.interaction.register_callback('place', self.place)
        self.interaction.register_callback('scale', self.scale)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        
    def main_loop(self):
        glutMainLoop()
        
if __name__ == "__main__":
    viewer = Viewer()
    viewer.main_loop()
        
