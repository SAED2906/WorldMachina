#!/usr/bin/env python3

import logging
import time
import math
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from render.skybox import Skybox
from render.planet import Planet
from render.shader_manager import ShaderManager
from ui.selector import TextureSelector


class Window:
    
    def __init__(self, engine):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Window initializing...")
        
        self.engine = engine
        self.title = "Planet Viewer"
        self.running = False
        self.last_frame_time = 0
        self.fullscreen = False
        
        pygame.init()
        
        display_info = pygame.display.Info()
        self.width = int(display_info.current_w * 0.8)
        self.height = int(display_info.current_h * 0.8)
        
        self.show_selector = False
        self.selected_skybox = None
        self.celestial_coordinates = {
            "ra": 0.0,
            "dec": 0.0
        }
        
        self.camera_distance = 5.0
        self.camera_rotation_x = 30.0
        self.camera_rotation_y = 0.0
        
        self._init_window()
        
        self.shader_manager = ShaderManager("src/shaders")
        self.shader_manager.load_shader_from_files("height_map")
        self.shader_manager.load_shader_from_files("ocean")
        
        self.shader_supported = self.shader_manager.check_shader_support()

        self.planet = Planet(radius=1.0, detail=320000)
        self.skybox = Skybox()
        self.texture_selector = TextureSelector(self.width, self.height)
        
        self.ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        try:
            self.planet.load_height_map("src/textures/planet/output4.png")
            self.planet.set_bump_strength(0.2)
            self.planet.set_height_scale(0.05)
            self.planet.use_height_map = False
        except Exception as e:
            self.logger.warning(f"Could not load planet maps: {e}")
        
        self.logger.info("Window initialized successfully")
    
    def _init_window(self):
        self.logger.info(f"Creating window: {self.title} ({self.width}x{self.height})")
        
        display_flags = DOUBLEBUF | OPENGL | RESIZABLE
        if self.fullscreen:
            display_flags |= FULLSCREEN
        
        self.screen = pygame.display.set_mode((self.width, self.height), display_flags)
        pygame.display.set_caption(self.title)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_TEXTURE_2D)

        print('[GL Blend] line: 135')

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 10, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        
        self._update_viewport()
        
        pygame.font.init()
        try:
            self.font = pygame.font.SysFont('Arial', 18)
        except:
            self.font = pygame.font.Font(None, 24)
        
        self.engine.register_component("window", self)
    
    def _update_viewport(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        self.ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
    
    def run(self):
        self.logger.info("Starting window main loop")
        self.running = True
        self.last_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            self._process_events()
            
            self.engine.update(delta_time)
            self._update_animation(delta_time)
            
            self._render_3d_scene()
            self._render_ui()
            
            pygame.display.flip()
            
            pygame.time.wait(16)
    
    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), 
                                                    DOUBLEBUF | OPENGL | RESIZABLE)
                self._update_viewport()
                self.texture_selector.resize(self.width, self.height)
                self.logger.info(f"Window resized to {self.width}x{self.height}")
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_s:
                    self.show_selector = not self.show_selector
                elif event.key == pygame.K_n:
                    self.planet.use_normal_map = not self.planet.use_normal_map
                    self.logger.info(f"Normal mapping: {'enabled' if self.planet.use_normal_map else 'disabled'}")
                elif event.key == pygame.K_b:
                    self.planet.use_bump_map = not self.planet.use_bump_map
                    self.logger.info(f"Bump mapping: {'enabled' if self.planet.use_bump_map else 'disabled'}")
                elif event.key == pygame.K_h:
                    self.planet.use_height_map = not self.planet.use_height_map
                    self.logger.info(f"Height mapping: {'enabled' if self.planet.use_height_map else 'disabled'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.planet.set_bump_strength(self.planet.bump_strength + 0.05)
                    self.logger.info(f"Bump strength: {self.planet.bump_strength:.2f}")
                elif event.key == pygame.K_MINUS:
                    self.planet.set_bump_strength(self.planet.bump_strength - 0.05)
                    self.logger.info(f"Bump strength: {self.planet.bump_strength:.2f}")
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.planet.set_height_scale(self.planet.height_scale + 0.01)
                    self.logger.info(f"Height scale: {self.planet.height_scale:.2f}")
                elif event.key == pygame.K_LEFTBRACKET:
                    self.planet.set_height_scale(self.planet.height_scale - 0.01)
                    self.logger.info(f"Height scale: {self.planet.height_scale:.2f}")
                
                elif event.key == pygame.K_UP:
                    self.planet.set_sea_level(self.planet.sea_level + 0.05)
                    self.logger.info(f"Sea level: {self.planet.sea_level:.2f}")
                elif event.key == pygame.K_DOWN:
                    self.planet.set_sea_level(self.planet.sea_level - 0.05)
                    self.logger.info(f"Sea level: {self.planet.sea_level:.2f}")

                elif event.key == pygame.K_o:
                    self.planet.set_ocean_radius(self.planet.ocean_radius + 0.001)
                    self.logger.info(f"Ocean radius: {self.planet.ocean_radius:.2f}")
                elif event.key == pygame.K_p:
                    self.planet.set_ocean_radius(self.planet.ocean_radius - 0.001)
                    self.logger.info(f"Ocean radius: {self.planet.ocean_radius:.2f}")
            
            if self.show_selector:
                if self.texture_selector.handle_event(event):
                    selected_texture, coordinates = self.texture_selector.get_selection()
                    if selected_texture:
                        self.selected_skybox = selected_texture
                        self.celestial_coordinates = coordinates
                        self.skybox.load_texture(selected_texture, coordinates)
                    continue
                
            elif event.type == pygame.MOUSEMOTION and not self.show_selector:
                if pygame.mouse.get_pressed()[0]:
                    dx, dy = event.rel
                    self.camera_rotation_y += dx * 0.5
                    self.camera_rotation_x += dy * 0.5
                    self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.show_selector:
                if event.button == 4:
                    self.camera_distance = max(0.1, self.camera_distance - 0.01)
                elif event.button == 5:
                    self.camera_distance = min(20.0, self.camera_distance + 0.5)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        
        if not self.fullscreen:
            self.windowed_width = self.width
            self.windowed_height = self.height
        
        if self.fullscreen:
            display_info = pygame.display.Info()
            self.width = display_info.current_w
            self.height = display_info.current_h
        else:
            if hasattr(self, 'windowed_width') and hasattr(self, 'windowed_height'):
                self.width = self.windowed_width
                self.height = self.windowed_height
            else:
                display_info = pygame.display.Info()
                self.width = int(display_info.current_w * 0.8)
                self.height = int(display_info.current_h * 0.8)
        
        display_flags = DOUBLEBUF | OPENGL
        if not self.fullscreen:
            display_flags |= RESIZABLE
        else:
            display_flags |= FULLSCREEN
        
        self.screen = pygame.display.set_mode((self.width, self.height), display_flags)
        
        self._update_viewport()
        
        self.texture_selector.resize(self.width, self.height)
        
        self.logger.info(f"Changed display mode: {self.width}x{self.height}, Fullscreen: {self.fullscreen}")
    
    def _update_animation(self, delta_time):
        self.planet.update(delta_time)
    
    def _render_3d_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        x_rad = math.radians(self.camera_rotation_x)
        y_rad = math.radians(self.camera_rotation_y)
        
        camera_x = self.camera_distance * math.cos(y_rad) * math.cos(x_rad)
        camera_y = self.camera_distance * math.sin(x_rad)
        camera_z = self.camera_distance * math.sin(y_rad) * math.cos(x_rad)
        
        gluLookAt(
            camera_x, camera_y, camera_z,
            0, 0, 0,
            0, 1, 0
        )
        
        self.skybox.render()
        
        self.planet.render(self.shader_manager)
        
        if not self.selected_skybox:
            self._draw_grid()
    
    def _render_ui(self):
        self.ui_surface.fill((0, 0, 0, 0))
        
        self._draw_ui_panels()
        
        self._draw_ui_text()
        
        if self.show_selector:
            self.texture_selector.render(self.ui_surface)
        
        screen_buffer = pygame.image.tostring(self.ui_surface, "RGBA", True)
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, 
                     GL_RGBA, GL_UNSIGNED_BYTE, screen_buffer)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 0); glVertex2f(0, self.height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDeleteTextures(1, [texture_id])
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        glMatrixMode(GL_MODELVIEW)
    
    def _draw_ui_panels(self):
        left_panel_width = 250
        left_panel = pygame.Surface((left_panel_width, self.height), pygame.SRCALPHA)
        left_panel.fill((0, 0, 0, 128))
        self.ui_surface.blit(left_panel, (0, 0))
        
        right_panel_width = 250
        right_panel_x = self.width - right_panel_width
        right_panel = pygame.Surface((right_panel_width, self.height), pygame.SRCALPHA)
        right_panel.fill((0, 0, 0, 128))
        self.ui_surface.blit(right_panel, (right_panel_x, 0))
    
    def _draw_ui_text(self):
        left_panel_lines = [
            "CONTROLS:",
            "",
            "Left mouse + drag: Rotate camera",
            "Mouse wheel: Zoom in/out",
            "S key: Skybox selector",
            "F11: Toggle fullscreen",
            "ESC: Quit",
            "",
            "SHADER CONTROLS:",
            "N key: Toggle normal mapping",
            "B key: Toggle bump mapping",
            "H key: Toggle height mapping",
            "+/- keys: Adjust bump strength",
            "[/] keys: Adjust height scale",
            "",
            "OCEAN CONTROLS:",
            "UP/DOWN arrows: Adjust sea level",
            "O/P keys: Adjust ocean radius"
        ]
        
        y_offset = 30
        for line in left_panel_lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.ui_surface.blit(text_surface, (20, y_offset))
            y_offset += 24
        
        right_panel_lines = ["PLANET INFO:", ""]
        
        if self.shader_manager.glsl_supported:
            right_panel_lines.extend([
                "Shaders: Supported",
                f"Height mapping: {'ON' if self.planet.use_height_map else 'OFF'}",
                f"Height scale: {self.planet.height_scale:.2f}",
                f"Sea level: {self.planet.sea_level:.2f}",
                f"Ocean radius: {self.planet.ocean_radius:.2f}"
            ])
        else:
            right_panel_lines.append("Shaders: Not supported")
            
        right_panel_lines.append("")
        
        if self.selected_skybox:
            right_panel_lines.extend([
                f"Skybox: {os.path.basename(self.selected_skybox)}",
                f"RA: {self.celestial_coordinates['ra']:.2f} hours",
                f"Dec: {self.celestial_coordinates['dec']:.2f} degrees"
            ])
        else:
            right_panel_lines.extend([
                "No skybox selected",
                "Press 'S' to open selector"
            ])
        
        right_panel_lines.extend([
            "",
            f"Camera distance: {self.camera_distance:.1f}",
            f"Rotation X: {self.camera_rotation_x:.1f}°",
            f"Rotation Y: {self.camera_rotation_y:.1f}°"
        ])
        
        right_panel_x = self.width - 250 + 20
        y_offset = 30
        for line in right_panel_lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.ui_surface.blit(text_surface, (right_panel_x, y_offset))
            y_offset += 24

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        
        grid_size = 10
        grid_step = 1
        
        glBegin(GL_LINES)
        glColor3f(0.3, 0.3, 0.3)
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, 0, -grid_size)
            glVertex3f(i, 0, grid_size)
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(-grid_size, 0, i)
            glVertex3f(grid_size, 0, i)
        
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def cleanup(self):
        self.logger.info("Window cleaning up resources")
        
        if hasattr(self, 'planet'):
            self.planet.cleanup()
        
        pygame.quit()