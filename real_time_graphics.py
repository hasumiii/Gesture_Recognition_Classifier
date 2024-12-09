#pip install opencv-python pyglet imgui[full] PyOpenGL

#check https://pyimgui.readthedocs.io/en/latest/reference/imgui.core.html for imgui functions

import cv2
import pyglet
from pyglet.gl import *
import imgui
from imgui.integrations.pyglet import create_renderer
from datetime import datetime
import numpy as np
from FBO import *
from CLIP_helper import *


w = 1024
h = 1024


#_________________________________initialization
# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Create a Pyglet window
config = pyglet.gl.Config(sample_buffers=1, samples=4) # pyglet.gl.Config(double_buffer=True)
config.major_version = 4
window = pyglet.window.Window(w, h, "Webcam Viewer", config=config)
# Initialize the UI system
imgui.create_context()
impl = create_renderer(window) #PygletRenderer(window)
canvas = FBO(w, h)
#_________________________________initialization end

#_________________________________initialize CLIP
happy_vec = encode_text("happy")
sad_vec = encode_text("sad")
#_________________________________initialize CLIP end

#_________________________________variables


#parameters that we can access in the ui
class Params:
    def __init__(self):
        self.last_frame = None
        self.current_frame = None #current video frame unprocessed
        self.current_frame_texture = None
        self.proc_frame = None #current video frame processed
        self.proc_frame_texture = None
        self.graphics_frame = None #current frame with shapes and colors
        self.canvas_sprite = None

        self.frame_index = 0

        self.move_x = 0.0 #centroid of the movement
        self.move_y = 0.0

        self.rotation = 45.0
        self.img_color = (0.5, 1.0, 1.0)
        self.color = (1.0, 0, 0, 0.5)
        self.color2 = (1.0, 1.0, 0, 0.5)
        self.rect_length = 20.0
        self.line_x2 = 100.0

p = Params()


def process_video():
    #proc_frame = cv2.absdiff(current_frame, last_frame)
    p.proc_frame = cv2.cvtColor(p.current_frame, cv2.COLOR_BGR2GRAY)
    p.proc_frame = 255 - cv2.Canny(p.proc_frame, 100, 200)
    p.proc_frame = cv2.cvtColor(p.proc_frame, cv2.COLOR_GRAY2BGR)


#_________________________________draw shapes

def draw_shapes():
    p.frame_index += 1
    #_________________________________UI
    #here you can add UI elements like sliders and labels that adjust the behavior of the app   

    imgui.text("parameters")
    changed, p.rotation = imgui.slider_float("Rotation", p.rotation, 0.0, 360.0)
    changed, p.img_color  = imgui.color_edit3("Img Color", *p.img_color)
    changed, p.color = imgui.color_edit4("Color", *p.color)
    changed, p.color2 = imgui.color_edit4("Color2", *p.color2)     
    changed, p.line_x2 = imgui.slider_float("Line X2", p.line_x2, 0.0, 512.0)

    if imgui.button("Capture Frame"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"current_frame_{timestamp}.png", p.current_frame)
        cv2.imwrite(f"proc_frame_{timestamp}.png", p.proc_frame)
        cv2.imwrite(f"graphics_frame_{timestamp}.png", canvas.to_cv2())

    

    #_________________________________ML analysis    
    #here you can do any analysis you need to using ML models
    vec = encode_image(p.current_frame) #encode the raw video frame
    happiness = (np.dot(happy_vec, vec) - np.dot(sad_vec, vec))*100.0  
    imgui.text(f"happiness: ({happiness:.2f}%)")

    #_________________________________movement analysis
    #here you can do any OpenCV analysis to extract movement from the video
    current = cv2.cvtColor(p.current_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    last = cv2.cvtColor(p.last_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    diff = (current - last)**2

    #find centorid
    xx = np.linspace(0, 1, current.shape[1])
    yy = np.linspace(0, 1, current.shape[0])
    X, Y = np.meshgrid(xx, yy)

    sum = np.sum(diff)

    imgui.text(f"sum: {sum:.2f}")
    
    if sum>5.0:
        x = np.sum(X*diff) / sum
        y = np.sum(Y*diff) / sum
  

        x = x.item()*w
        y = h - y.item()*h

        p.move_x = p.move_x*0.95 + x*0.05
        p.move_y = p.move_y*0.95 + y*0.05
       

    #___________________________drawing background    
    #glClearColor(0.5, 0.5, 0.5, 0.0)
    #glClear(GL_COLOR_BUFFER_BIT)

    #this code makes the graphics overlay a bit faded every 100 frames therefore if nothing happens
    #for a while the graphics will fade away
    if p.frame_index % 100 == 0:
        fade_rect = pyglet.shapes.Rectangle(0, 0, w, h, color=(255, 255, 255))
        fade_rect.opacity = 2
        fade_rect.draw()
    

    #____________________________drawing shapes
    #here is the main drawing logic

    red = (happiness*200.0).clip(0, 255)
    blue = (-happiness*200.0).clip(0, 255)
    circle = pyglet.shapes.Circle(p.move_x, p.move_y, 20.0, 30, (int(red), int(blue), int(blue)))
    circle.opacity = 255
    circle.draw()

    pyglet.gl.glLineWidth(5)
    arc = pyglet.shapes.Arc(p.move_x, p.move_y, 20.0, 30, color=(0, 0, 0), thickness=1.0)
    arc.draw()


    
            
#_________________________________draw on main window


def draw_on_window():
    w2 = w//2
    h2 = h//2

    if p.canvas_sprite is None:
        p.canvas_sprite = pyglet.sprite.Sprite(canvas.texture)
        p.canvas_sprite.scale = 0.5
        p.canvas_sprite.x = w2
        p.canvas_sprite.y = h2
        p.canvas_sprite.blend_mode = pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA

    p.canvas_sprite.image = p.proc_frame_texture    
    p.canvas_sprite.x = 0.0
    p.canvas_sprite.y = 0.0
    p.canvas_sprite.width = w
    p.canvas_sprite.height = h
    p.canvas_sprite.draw()

    p.canvas_sprite.image = canvas.texture
    p.canvas_sprite.x = 0.0
    p.canvas_sprite.y = 0.0
    p.canvas_sprite.width = w
    p.canvas_sprite.height = h
    p.canvas_sprite.draw()


#_________________________________IGNORE BELOW THIS LINE


#_________________________________update

def update(dt):
    ret, frame = cap.read()
    if ret:
        p.last_frame = p.current_frame
        #center crop and convert video frame to texture
        h, w, _ = frame.shape
        if h > w:
            frame = frame[(h-w)//2:(h+w)//2, :]
        else:
            frame = frame[:, (w-h)//2:(w+h)//2]

        p.current_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

        if p.last_frame is None:
            p.last_frame = p.current_frame

        process_video()  

        frame_rgb = cv2.cvtColor(p.current_frame, cv2.COLOR_BGR2RGB)        
        texture_frame_rgb = cv2.flip(frame_rgb, 0)
        height, width, channels = texture_frame_rgb.shape

        if p.current_frame_texture is None:            
            p.current_frame_texture = pyglet.image.ImageData(width, height, 'RGB', texture_frame_rgb.tobytes())
        else:
            p.current_frame_texture.set_data('RGB', width*3, texture_frame_rgb.tobytes())

        proc_frame_rgb = cv2.cvtColor(p.proc_frame, cv2.COLOR_BGR2RGB)
        texture_proc_frame_rgb = cv2.flip(proc_frame_rgb, 0)
        height, width, channels = texture_proc_frame_rgb.shape

        if p.proc_frame_texture is None:
            p.proc_frame_texture = pyglet.image.ImageData(width, height, 'RGB', texture_proc_frame_rgb.tobytes())
        else:
            p.proc_frame_texture.set_data('RGB', width*3, texture_proc_frame_rgb.tobytes())
  

#_________________________________update end


#_________________________________draw
@window.event
def on_draw():
    global current_frame, current_frame_texture, proc_frame_texture, proc_frame, graphics_frame

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    window.clear()
    imgui.new_frame()

    imgui.begin("Graphics")    

    if p.current_frame is not None:
        #p.graphics_frame = canvas.to_cv2()  

        canvas.bind()
        draw_shapes()
        canvas.unbind(window)

        draw_on_window()
        
    imgui.end()
    imgui.render()
    impl.render(imgui.get_draw_data())
#_________________________________draw end

#_________________________________on_close
@window.event
def on_close():
    cap.release()
    impl.shutdown()
    pyglet.app.exit()
#_________________________________on_close end

#_________________________________run
pyglet.clock.schedule_interval(update, 1/20.0)
pyglet.app.run()
#_________________________________run end