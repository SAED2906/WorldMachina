#!/usr/bin/env python3

import logging
import os
import pygame
import math


class TextureSelector:
    
    def __init__(self, window_width, window_height):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing texture selector UI...")
        
        self.window_width = window_width
        self.window_height = window_height
        
        self.selected_texture = None
        self.celestial_coordinates = {
            "ra": 12.0,
            "dec": 0.0
        }
        self.drag_active = False
        self.drag_coordinate = None
        
        self.ui_padding = 20
        self.ui_width = min(800, window_width - 2 * self.ui_padding)
        self.ui_height = min(600, window_height - 2 * self.ui_padding)
        self.ui_x = (window_width - self.ui_width) // 2
        self.ui_y = (window_height - self.ui_height) // 2
        
        self._init_ui()
        
        self.textures = self._find_textures()
    
    def _init_ui(self):
        pygame.font.init()
        self.title_font = pygame.font.SysFont('Arial', 24)
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 14)
        
        self.ui_surface = pygame.Surface((self.ui_width, self.ui_height), pygame.SRCALPHA)
        
        self.bg_color = (0, 0, 0, 200)
        self.text_color = (255, 255, 255)
        self.highlight_color = (64, 128, 255)
        self.button_color = (40, 40, 40)
        self.button_hover_color = (60, 60, 60)
        
        self.header_height = 40
        self.texture_list_width = self.ui_width // 3
        self.texture_list_height = self.ui_height - self.header_height - 60
        self.coordinate_height = 60
        
        self.close_button = {
            "rect": pygame.Rect(self.ui_width - 40, 10, 30, 30),
            "text": "X",
            "action": "close"
        }
        
        slider_width = self.ui_width - self.texture_list_width - 3 * self.ui_padding
        self.ra_slider = {
            "rect": pygame.Rect(self.texture_list_width + 2 * self.ui_padding, 
                               self.ui_height - 50, 
                               slider_width, 20),
            "min": 0.0,
            "max": 24.0,
            "value": self.celestial_coordinates["ra"],
            "name": "Right Ascension (hours)",
            "key": "ra"
        }
        
        self.dec_slider = {
            "rect": pygame.Rect(self.texture_list_width + 2 * self.ui_padding, 
                               self.ui_height - 90, 
                               slider_width, 20),
            "min": -90.0,
            "max": 90.0,
            "value": self.celestial_coordinates["dec"],
            "name": "Declination (degrees)",
            "key": "dec"
        }
    
    def resize(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        
        self.ui_width = min(800, window_width - 2 * self.ui_padding)
        self.ui_height = min(600, window_height - 2 * self.ui_padding)
        self.ui_x = (window_width - self.ui_width) // 2
        self.ui_y = (window_height - self.ui_height) // 2
        
        self._init_ui()
    
    def _find_textures(self):
        texture_paths = []
        
        dirs_to_check = [
            "src/textures/skybox",
            "assets/textures/skybox",
            "skybox",
            "textures"
        ]
        
        for directory in dirs_to_check:
            if os.path.isdir(directory):
                for filename in os.listdir(directory):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tga')):
                        texture_paths.append(os.path.join(directory, filename))
                        
                if texture_paths:
                    break
        
        if not texture_paths:
            self.logger.warning("No skybox textures found. Checked directories: " + 
                               ", ".join(dirs_to_check))
            
        self.logger.info(f"Found {len(texture_paths)} skybox textures")
        return texture_paths
    
    def handle_event(self, event):
        if not hasattr(event, 'pos'):
            return False
            
        ui_x = event.pos[0] - self.ui_x
        ui_y = event.pos[1] - self.ui_y
        
        if (ui_x < 0 or ui_x >= self.ui_width or 
            ui_y < 0 or ui_y >= self.ui_height):
            if event.type == pygame.MOUSEBUTTONUP:
                self.drag_active = False
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.close_button["rect"].collidepoint(ui_x, ui_y):
                return True
            
            texture_list_rect = pygame.Rect(
                self.ui_padding, 
                self.header_height + self.ui_padding,
                self.texture_list_width,
                self.texture_list_height
            )
            
            if texture_list_rect.collidepoint(ui_x, ui_y):
                texture_index = (ui_y - texture_list_rect.y) // 30
                if 0 <= texture_index < len(self.textures):
                    self.selected_texture = self.textures[texture_index]
                return True
            
            if self.ra_slider["rect"].collidepoint(ui_x, ui_y):
                self.drag_active = True
                self.drag_coordinate = "ra"
                self._update_slider_value(self.ra_slider, ui_x)
                return True
                
            if self.dec_slider["rect"].collidepoint(ui_x, ui_y):
                self.drag_active = True
                self.drag_coordinate = "dec"
                self._update_slider_value(self.dec_slider, ui_x)
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self.drag_active and self.drag_coordinate:
                if self.drag_coordinate == "ra":
                    self._update_slider_value(self.ra_slider, ui_x)
                else:
                    self._update_slider_value(self.dec_slider, ui_x)
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag_active = False
            return True
        
        return False
    
    def _update_slider_value(self, slider, x_pos):
        rel_pos = max(0.0, min(1.0, (x_pos - slider["rect"].x) / slider["rect"].width))
        
        value_range = slider["max"] - slider["min"]
        value = slider["min"] + rel_pos * value_range
        
        slider["value"] = value
        
        self.celestial_coordinates[slider["key"]] = value
    
    def render(self, surface):
        self.ui_surface.fill((0, 0, 0, 0))
        
        pygame.draw.rect(self.ui_surface, self.bg_color, 
                        (0, 0, self.ui_width, self.ui_height), 0, 10)
        
        pygame.draw.rect(self.ui_surface, self.highlight_color, 
                        (0, 0, self.ui_width, self.ui_height), 2, 10)
        
        title_text = self.title_font.render("Skybox Selector", True, self.text_color)
        self.ui_surface.blit(title_text, (20, 10))
        
        pygame.draw.rect(self.ui_surface, self.button_color, 
                        self.close_button["rect"], 0, 5)
        close_text = self.font.render(self.close_button["text"], True, self.text_color)
        self.ui_surface.blit(close_text, (
            self.close_button["rect"].x + (self.close_button["rect"].width - close_text.get_width()) // 2,
            self.close_button["rect"].y + (self.close_button["rect"].height - close_text.get_height()) // 2
        ))
        
        texture_list_rect = pygame.Rect(
            self.ui_padding, 
            self.header_height + self.ui_padding,
            self.texture_list_width,
            self.texture_list_height
        )
        pygame.draw.rect(self.ui_surface, (30, 30, 30), texture_list_rect, 0, 5)
        
        for i, texture_path in enumerate(self.textures):
            item_rect = pygame.Rect(
                texture_list_rect.x,
                texture_list_rect.y + i * 30,
                texture_list_rect.width,
                30
            )
            
            if texture_path == self.selected_texture:
                pygame.draw.rect(self.ui_surface, self.highlight_color, item_rect, 0)
            
            texture_name = os.path.basename(texture_path)
            name_text = self.small_font.render(texture_name, True, self.text_color)
            self.ui_surface.blit(name_text, (item_rect.x + 10, item_rect.y + 8))
        
        if self.selected_texture:
            preview_rect = pygame.Rect(
                self.texture_list_width + self.ui_padding * 2, 
                self.header_height + self.ui_padding,
                self.ui_width - self.texture_list_width - self.ui_padding * 3,
                self.ui_height - self.header_height - self.coordinate_height - self.ui_padding * 2
            )
            pygame.draw.rect(self.ui_surface, (30, 30, 30), preview_rect, 0, 5)
            
            texture_name = os.path.basename(self.selected_texture)
            name_text = self.font.render(f"Selected: {texture_name}", True, self.text_color)
            self.ui_surface.blit(name_text, (
                preview_rect.x + 10,
                preview_rect.y + 10
            ))
            
            try:
                texture_img = pygame.image.load(self.selected_texture)
                aspect_ratio = texture_img.get_width() / texture_img.get_height()
                
                preview_width = preview_rect.width - 20
                preview_height = preview_rect.height - 40
                
                if aspect_ratio > 1:
                    target_width = preview_width
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_height = preview_height
                    target_width = int(target_height * aspect_ratio)
                
                scaled_img = pygame.transform.scale(texture_img, (target_width, target_height))
                
                self.ui_surface.blit(scaled_img, (
                    preview_rect.x + (preview_rect.width - target_width) // 2,
                    preview_rect.y + 40 + (preview_height - target_height) // 2
                ))
                
            except Exception as e:
                error_text = self.font.render(f"Preview unavailable: {e}", True, (255, 100, 100))
                self.ui_surface.blit(error_text, (
                    preview_rect.x + 10,
                    preview_rect.y + 50
                ))
        
        self._draw_slider(self.ra_slider)
        
        self._draw_slider(self.dec_slider)
        
        surface.blit(self.ui_surface, (self.ui_x, self.ui_y))
    
    def _draw_slider(self, slider):
        label_text = self.font.render(f"{slider['name']}: {slider['value']:.2f}", True, self.text_color)
        self.ui_surface.blit(label_text, (slider["rect"].x, slider["rect"].y - 25))
        
        pygame.draw.rect(self.ui_surface, (60, 60, 60), slider["rect"], 0, 5)
        
        value_range = slider["max"] - slider["min"]
        rel_pos = (slider["value"] - slider["min"]) / value_range
        handle_x = slider["rect"].x + int(rel_pos * slider["rect"].width)
        
        handle_rect = pygame.Rect(handle_x - 5, slider["rect"].y - 5, 10, slider["rect"].height + 10)
        pygame.draw.rect(self.ui_surface, self.highlight_color, handle_rect, 0, 5)
    
    def get_selection(self):
        return self.selected_texture, self.celestial_coordinates