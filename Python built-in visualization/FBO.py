import pyglet
from pyglet.gl import *
import numpy as np
import cv2


class FBO:
    def __init__(self, W, H):
        self.width = W
        self.height = H
        
        self.texture = pyglet.image.Texture.create(W, H, min_filter=GL_LINEAR, mag_filter=GL_LINEAR)
        #self.depth_buffer = pyglet.image.Renderbuffer(W, H, GL_DEPTH_COMPONENT)

        # Create a Framebuffer, and attach:
        self.framebuffer = pyglet.image.Framebuffer()
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        #self.framebuffer.attach_renderbuffer(self.depth_buffer, attachment=GL_DEPTH_ATTACHMENT)

        #self.batch = pyglet.graphics.Batch()
        #image = pyglet.resource.image('adriaen-brouwer_portrait-of-a-man.jpg')
        self.sprite = pyglet.sprite.Sprite(self.texture)

    def bind(self):
        self.framebuffer.bind()
        #glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        glViewport(0, 0, self.width, self.height)

    def unbind(self, window):
        self.framebuffer.unbind()
        #glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, window.width, window.height)


    def to_cv2(self):
        img_data = self.texture.get_image_data()
        img_bytes = img_data.get_bytes("RGBA", img_data.width * 4)
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_data.height, img_data.width, 4)
        #flip vertically
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img