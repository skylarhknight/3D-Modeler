"""
3D Modeller

"""

"""Import Packages"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

""" Viewer Class """
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
        glutInit()                                          # Initialize GLUT
        glutInitWindowSize(640, 480)                        # Window Size
        gluCreateWindow("3D Modeller")                      # Window Title
        gluInitDisplayMode(GLUT_SINGLE | GLUT_RGB)          # Display Mode
        gluDisplayFunc(self.render)                         # Register Render Function
        
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
        
        hierarchical_node = SnowFigure()
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
        
    # Render | the render pass for the scene
    def render(self):
        self.init_view()
        
        glEnable(GL_LIGHTING)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Load the model view matrix from the current state of the trackball
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()                                    # Reset the model view matrix
        loc = self.interaction.translation                  # Get the translation
        glTranslated(loc[0], loc[1], loc[2])                # Apply the translation
        glMultMatrixd(self.interaction.trackball.matrix)    # Apply the trackball rotation
        
        # Store the inverse of current model view matrix
        currentModelView = np.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        self.modelView = np.transpose(currentModelView)
        self.inverseModelView = inv(np.transpose(currentModelView))
        
        # Render scene (calls render func for each object)
        self.scene.render()
        
        # Draw Grid
        glDisable(GL_LIGHTING)
        glCallList(G_OBJ_PLANE)
        glPopMatrix()
        
        # Flush buffers so scene can be drawn
        glFlush()
        
    """ Initialize the projection matrix """
    def init_view(self):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        aspect_ration = float(xSize) / float(ySize)
        
        # load projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glLoadIdentity()
        
        
        glViewport(0, 0, xSize, ySize)
        gluPerspective(70, aspect_ratio, 0.1, 1000.0)
        glTranslated(0, 0, -15)
        
    def move(self, x, y):
        start, direction = self.get_ray(x, y)
        self.scene.move_selected(start, direction, self.inverseModelView)
        
    def rotate_color(self, forward):
        self.scene.rotate_selected_color(forward)
    
    def scale(self, up):
        self.scene.scale_selected(up)
    
    def place(self, shape, x, y):
        start, direction = self.get_ray(x, y)
        self.scene.place(shape, start, direction, self.inverseModelView)
    
        


class Scene(object):
    
    # default depth from the camera objects are placed at
    PLACE_DEPTH = 15.0
    
    def __init__(self):
        # scene keeps list of nodes that are displayed
        self.node_list = list()
        # Keep track of the current selected node
        # actions depend on the selected node
        self.selected_node = None
        
    # add a new node to the scene
    def add_node(self, node):
        self.node_list.append(node)
        
    # render the scene
    def render(self):
        for node in self.node_list:
            node.render()
    
    def get_ray(self, x, y):
        # generate ray at the near plane, in the direciton of x, y
        self.init_view()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # get two points on the line
        start = np.array(gluUnProject(x, y, 0.001))
        end   = np.array(gluUnProject(x, y, 0.999))
        
        # convert points to ray
        direction = end - start
        direction = direction / norm(direction)
        
        return (start, direction)
    
    # execute pick of object
    def pick(self, x, y):
        start, direction = self.get_ray(x, y)
        self.scene.pick(start, direction, self.modelView)
        
        
    # changing color
    def rotate_selected_color(self, forwards):
        if self.selected_node is None:
            return
        self.selected_node.rotate_color(forwards)
        
    # Move the selected node
    def move_selected(self, start, direction, inv_modelview):
        if self.selected_node is None:
            return
        
        node = self.selected_node
        depth = node.depth
        oldloc = node.selected_loc
        
        newloc = (start + direction * depth)
        
        translation = newloc - oldloc
        pre_tran = np.array([translation[0], translation[1], translation[2], 0])
        translation = inv_modelview.dot(pre_tran)
        
        node.translate(translation[0], translation[1], translation[2])           # Translate the node
        node.selected_loc = newloc                                               # Update the selected location
        
    def place(self, shape, start, direction, inv_modelview):
        new_node = None
        if shape == 'cube':
            new_node = Cube()
        elif shape == 'sphere':
            new_node = Sphere()
        elif shape == 'snow_figure':
            new_node = SnowFigure()
            
        self.add_node(new_node)
        
        translation = (start + direction * self.PLACE_DEPTH)
        
        pre_tran = np.array([translation[0], translation[1], translation[2], 1])
        translation = inv_modelview.dot(pre_tran)
        
        new_node.translate(translation[0], translation[1], translation[2])
            
            
            
""" Base class for scene elements """
class Node(object):
    
    def __init__(self):
        # Initialize the transformation matrix
        self.selected = False
        self.color_index = random.randint(color.MIN_COLOR, color.MAX_COLOR)     # Random Color
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])                      # Axis Aligned Bounding Box
        self.scaling_matrix = np.identity(4)
        self.translation_matrix = np.identity(4)
    
    # Render items to screen
    def render(self):
        glPushMatrix()                                                      # Push Matrix
        glMultMatrixxf(np.transpose(self.translation_matrix))               # Apply Translation
        glMultMatrixf(self.scaling_matrix)                                  # Apply Scaling
        cur_color = color.COLORS[self.color_index]                          # Get Color
        glColor3f(cur_color[0], cur_color[1], cur_color[2])                 # Set Color
        
        # emit light if the object (node) is selected
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3])    
            
        self.render_self()                                                  # Render the object
        
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()                                                       # Pop Matri
        
        # Changing colors
        def rotate_color(self, forward):
            self.color_index += 1 if forward else -1
            if self.color_index > color.MAX_COLOR:          # creates loop between max and min colors
                self.color_index = color.MIN_COLOR
            if self.color_index < color.MIN_COLOR:
                self.color_index = color.MAX_COLOR
                
        # Scaling Nodes
        def scale(self, up):
            s = 1.1 if up else 0.9
            self.scaling_matrix = np.dot(self.scaling_matrix, scaling([s, s, s]))
            self.aabb.scale(s)
            
            def scaling(scale):
                s = numpy.identity(4)
                s[0, 0] = scale[0]
                s[1, 1] = scale[1]
                s[2, 2] = scale[2]
                s[3, 3] = 1
                return s
        
        
    def render_self(self):
        raise NotImplementedError("The Abstract Node Class doesn't define 'render_self'")
    
    def pick(self, start, direction, mat):
        # check if the ray intersects the bounding box
        newmat = np.dot(
            np.dot(mat, self.translation_matrix),
            np.linalg.inv(self.scaling_matrix)
        )
        results = self.aabb.ray_hit(start, direction, newmat)
        return results
    
    def select(self, select = None):
        if select is not None:
            self.selected = select 
        else:
            self.selected = not self.selected
    
    # translate node
    def translate(self, x, y, z):
        self.translation_matrix = np.dot(
            self.translation_matrix, 
            translation([x, y, z]))
        
        def translation(displacement):
            t = numpy.identity(4)
            t[0, 3] = displacement[0]
            t[1, 3] = displacement[1]
            t[2, 3] = displacement[2]
            return t
            
    
    
""" Primitive Class """
class Primitive(Node):
    def __init__(self):
        super(Primitive, self).__init__()
        self.call_list = None
        
    def render_self(self):
        glCallList(self.call_list)

class Cube(Primitive):
    def __init__(self):
        super(Cube, self).__init__()
        self.call_list = G_OBJ_CUBE   
    
class Sphere(Primitive):
    def __init__(self):
        super(Sphere, self).__init__()
        self.call_list = G_OBJ_SPHERE
        

class HierarchicalNode(Node):
    def __init__(self):
        super(HierarchicalNode, self).__init__()
        self.child_nodes = []
        
    def render_self(self):
        for child in self.child_nodes:
            child.render()
            

"""" Interaction Class """
class Interaction(object):
    self.pressed = None                                 # Currently pressed mouse button
    self.mouse_loc = None                               # Current mouse location   
    self.translation = [0, 0, 0, 0]                     # current location of camera
    self.trackball = trackball.Trackball(theta = -25, distance = 15)  # Trackball for rotation
    self.callbacks = defaultdict(list)                  # Callbacks for different actions
    
    self.register()
    
    # Register the callbacks
    def register(self):
        glutMouseFunc(self.handle_mouse_button)         # Mouse Button Callback
        glutMotionFunc(self.handle_mouse_move)          # Mouse Motion Callback
        glutKeyboardFunc(self.handle_keystroke)         # Keyboard Callback
        
    def translate(self, x, y, z):
        self.translation = [x, y, z, 0]
        self.translation[0] += x
        self.translation[1] += y
        self.translation[2] += z
        
    # called when mouse button is pressed or released
    def handle_mouse_button(self, button, mode, x, y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - y
        self.mouse_loc = (x, y)
        
        if mode == GLUT_DOWN:                       # Mouse Button Pressed
            self.pressed = button
            if button == GLUT_RIGHT_BUTTON: 
               pass
            elif button == GLUT_LEFT_BUTTON:         # Pick
               self.callbacks('pick', x, y)
            elif button == 3:                        # Scroll Up
               self.translate(0, 0, 1.0)
            elif button == 4:                        # Scroll Down
                self.translate(0, 0, -1.0)
        else:                                       # Mouse Button Released
            self.pressed = None
        glutPostRedisplay()
        
    def handle_mouse_move(self, x, screen_y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - screen_y
        
        if self.pressed is not None:
            dx = x - self.mouse_loc[0]
            dy = y - self.mouse_loc[1]
            
            if self.pressed == GLUT_RIGHT_BUTTON and self.trackball is not None:
                self.trackball.drag_to(self.mouse_loc[0], self.mouse_loc[1], dx, dy)
            elif self.pressed == GLUT_LEFT_BUTTON:
                self.trigger('move', x, y)
            elif self.pressed == GLUT_MIDDLE_BUTTON:
                self.translate(dx/60,0, dy/60.0, 0)
            else:
                pass
            glutPostRedisplay()
        self.mouse_loc = (x, y)
    
    # handle keyboard input
    def handle_keystroke(self, key, x, screen_y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - screen_y
        if key == 'c':
            self.trigger('place', 'cube', x, y)
        elif key == 's':
            self.trigger('place', 'sphere', x, y)
        elif key == GLUT_KEY_UP:
            self.trigger('scale', up = True)
        elif key == GLUT_KEY_DOWN:
            self.trigger('scale', up = False)
        elif key == GLUT_KEY_LEFT:
            self.trigger('rotate_color', forward = True)
        elif key == GLUT_KEY_RIGHT:
            self.trigger('rotate_color', forward = False)
        glutPostRedisplay()
        
    def register_callback(self, name, func):
        self.callbacks[name].append(func)
    
    def trigger(self, name, *args, **kwargs):
        for func in self.callbacks[name]:
            func(*args, **kwargs)
            
    # execute selection
    def pick(self, start, direction, mat):
        if self.selected_node is not None:
            self.selected_node.select = False
            self.selected_node = None
        
        # keep track of the closest object
        mindist = sys.maxint
        closest_node = None
        for node in self.node_list:
            hit, distance = node.pick(start, direction, mat)
            if hit and distance < mindist:
                mindist, closest_node = distance, node
                
        # keep track if we hit someting
        if closest_node is not None:
            closest_node.select()
            closest_node.depth = mindist
            closest_node.sleected_loc = start + direction * mindist
            self.selected_node = closest_node
        
class SnowFigure(HierarchicalNode):
    def __init__(self):
        super(SnowFigure, self).__init__()
        self.child_nodes = [Sphere(), Sphere(), Sphere()]
        self.child_nodes[0].translate(0, -0.6, 0) # scale 1.0
        self.child_nodes[1].translate(0, 0.1, 0)
        self.child_nodes[1].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.8, 0.8, 0.8]))
        self.child_nodes[2].translate(0, 0.75, 0)
        self.child_nodes[2].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.7, 0.7, 0.7]))
        for child_node in self.child_nodes:
            child_node.color_index = color.MIN_COLOR
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 1.1, 0.5])