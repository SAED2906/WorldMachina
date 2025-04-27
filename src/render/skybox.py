#!/usr/bin/env python3

import logging
import math
import os
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *


class Skybox:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing skybox...")
        
        self.texture_id = None
        self.texture_path = None
        self.celestial_coordinates = {
            "ra": 0.0,
            "dec": 0.0
        }
    
    def load_texture(self, texture_path, coordinates=None):
        if coordinates:
            self.celestial_coordinates = coordinates
        
        if self.texture_path == texture_path and self.texture_id:
            return True
            
        self.texture_path = texture_path
        
        try:
            if self.texture_id:
                try:
                    glDeleteTextures(self.texture_id)
                except:
                    pass
            
            if not os.path.isfile(texture_path):
                self.logger.error(f"Texture file not found: {texture_path}")
                self.texture_id = None
                return False
            
            textureSurface = pygame.image.load(texture_path)
            textureData = pygame.image.tostring(textureSurface, "RGBA", 1)
            
            width = textureSurface.get_width()
            height = textureSurface.get_height()
            
            self.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                         GL_RGBA, GL_UNSIGNED_BYTE, textureData)
                         
            self.logger.info(f"Loaded skybox texture: {texture_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load skybox texture {texture_path}: {e}")
            self.texture_id = None
            return False
    
    def render(self):
        if not self.texture_id:
            return
        
        glPushAttrib(GL_ENABLE_BIT)
        glPushMatrix()
        
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glEnable(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        radius = 50.0
        
        ra_degrees = self.celestial_coordinates["ra"] * 15.0
        glRotatef(-90.0, 1.0, 0.0, 0.0)
        glRotatef(self.celestial_coordinates["dec"], 1.0, 0.0, 0.0)
        glRotatef(ra_degrees, 0.0, 0.0, 1.0)
        
        glFrontFace(GL_CW)
        
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluQuadricTexture(quadric, GL_TRUE)
        
        gluSphere(quadric, radius, 32, 32)
        
        gluDeleteQuadric(quadric)
        glFrontFace(GL_CCW)
        
        glDepthMask(GL_TRUE)
        glPopMatrix()
        glPopAttrib()

    def cleanup(self):
        if self.texture_id:
            try:
                glDeleteTextures(self.texture_id)
            except:
                pass