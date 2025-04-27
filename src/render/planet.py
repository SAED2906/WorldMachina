#!/usr/bin/env python3

import logging
import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *


class Planet:
    
    def __init__(self, radius=1.0, detail=32):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing planet object...")
        
        self.radius = radius
        self.detail = detail
        self.rotation = 0.0
        self.rotation_speed = 0.0
        
        self.diffuse_texture = None
        self.height_texture = None
        
        self.use_height_map = False
        self.height_scale = 0.5
        
        self.sea_level = 0.3
        self.ocean_radius = 0.97
        self.use_bump_map = False
        self.bump_strength = 0.2
        
        try:
            self._load_diffuse_texture("src/textures/planet/output6.png")
        except Exception as e:
            self.logger.warning(f"Could not load default planet texture: {e}")
    
    def create_debug_heightmap_sphere(self):
        print("Creating debug height-mapped sphere...")
        
        if not self.height_texture:
            print("Error: No height map loaded for debug sphere")
            return
        
        from OpenGL.arrays import vbo
        import numpy as np
        
        vertices = []
        normals = []
        texcoords = []
        
        try:
            import pygame
            glBindTexture(GL_TEXTURE_2D, self.height_texture)
            width = 64
            height = 64
            pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
            height_data = np.frombuffer(pixels, dtype=np.float32).reshape(width, height)
            
            print(f"Debug height map data loaded: {height_data.shape}")
            print(f"Height range: min={height_data.min()}, max={height_data.max()}")
            
        except Exception as e:
            print(f"Debug sphere creation failed: {e}")

    def validate_geometry(self):
        expected_vertices = (self.detail + 1) * (self.detail + 1)
        
        print(f"Planet geometry validation:")
        print(f"- Sphere detail level: {self.detail}")
        print(f"- Expected vertex count: ~{expected_vertices}")
        print(f"- Height mapping: {'ENABLED' if self.use_height_map else 'DISABLED'}")
        print(f"- Current height scale: {self.height_scale:.2f}")
        
        if self.detail < 64:
            print("WARNING: Sphere detail may be too low for visible displacement")
            print("Recommend increasing to 64+ for better results")
        
        return expected_vertices


    def _load_diffuse_texture(self, texture_path):
        try:
            textureSurface = pygame.image.load(texture_path)
            textureData = pygame.image.tostring(textureSurface, "RGBA", 1)
            
            width = textureSurface.get_width()
            height = textureSurface.get_height()
            
            if self.diffuse_texture:
                glDeleteTextures(self.diffuse_texture)
            
            self.diffuse_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                         GL_RGBA, GL_UNSIGNED_BYTE, textureData)
            
            glGenerateMipmap(GL_TEXTURE_2D)
                         
            self.logger.info(f"Loaded planet diffuse texture: {texture_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load texture {texture_path}: {e}")
            self.diffuse_texture = None
            return False
    
    def load_height_map(self, texture_path):
        try:
            textureSurface = pygame.image.load(texture_path)
            textureData = pygame.image.tostring(textureSurface, "RGBA", 1)
            
            width = textureSurface.get_width()
            height = textureSurface.get_height()
            
            if self.height_texture:
                glDeleteTextures(self.height_texture)
            
            self.height_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.height_texture)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                         GL_RGBA, GL_UNSIGNED_BYTE, textureData)
            
            glGenerateMipmap(GL_TEXTURE_2D)
            
            self.height_texture = self.height_texture
            self.use_height_map = True
                         
            self.logger.info(f"Loaded planet height map: {texture_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load height map {texture_path}: {e}")
            self.height_texture = None
            self.use_height_map = False
            return False
    
    def set_height_scale(self, scale):
        self.height_scale = scale
    
    def set_bump_strength(self, strength):
        self.bump_strength = max(0.0, min(1.0, strength))
        self.logger.info(f"Bump strength set to: {self.bump_strength:.2f}")
    
    def set_sea_level(self, level):
        self.sea_level = max(0.0, min(10.0, level))
        self.logger.info(f"Sea level set to: {self.sea_level:.2f}")
    
    def set_ocean_radius(self, radius_factor):
        self.ocean_radius = max(0.8, min(10.0, radius_factor))
        self.logger.info(f"Ocean radius set to: {self.ocean_radius:.2f} of planet radius")
    
    def update(self, delta_time):
        self.rotation += self.rotation_speed * delta_time
        if self.rotation >= 360.0:
            self.rotation -= 360.0

    def render(self, shader_manager=None):
        glPushMatrix()
        glRotatef(self.rotation, 0, 1, 0)
        glRotatef(90, -1, 0, 0)

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluQuadricTexture(quadric, GL_TRUE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        if shader_manager and shader_manager.glsl_supported and self.diffuse_texture:
            if self.use_height_map and self.height_texture:
                if shader_manager.use_shader("height_map"):
                    shader_manager.set_uniform_bool("useHeightMap", self.use_height_map)
                    shader_manager.set_uniform_float("heightScale", self.height_scale)
                    
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, self.height_texture)
                    
                    gluSphere(quadric, self.radius, self.detail, self.detail)
                    
                    shader_manager.stop_using_shader()
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, 0)
        else:
            if self.diffuse_texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
                glColor3f(1.0, 1.0, 1.0)
            else:
                glDisable(GL_TEXTURE_2D)
                glColor3f(0.2, 0.6, 0.8)
            
            gluSphere(quadric, self.radius, self.detail, self.detail)
            glDisable(GL_TEXTURE_2D)
        
        if shader_manager and shader_manager.glsl_supported and self.diffuse_texture:
            if self.use_height_map and self.height_texture:
                if shader_manager.use_shader("ocean"):
                    glDepthMask(GL_FALSE)
                    
                    shader_manager.set_uniform_bool("useHeightMap", self.use_height_map)
                    shader_manager.set_uniform_float("heightScale", self.height_scale)
                    shader_manager.set_uniform_float("seaLevel", self.sea_level)
                    
                    shader_manager.set_uniform_vec3("oceanColor", 0.0, 0.3, 0.6)
                    shader_manager.set_uniform_vec3("shoreColor", 0.9, 0.8, 0.6)
                    
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, self.height_texture)
                    
                    gluSphere(quadric, self.radius * self.ocean_radius, self.detail, self.detail)
                    
                    shader_manager.stop_using_shader()
                    glDepthMask(GL_TRUE)
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, 0)
        
        glDisable(GL_BLEND)
        
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def cleanup(self):
        if self.diffuse_texture:
            glDeleteTextures([self.diffuse_texture])
            
        if self.height_texture:
            glDeleteTextures([self.height_texture])