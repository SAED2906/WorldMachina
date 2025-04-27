#!/usr/bin/env python3

import os
import logging
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

class ShaderManager:
    def __init__(self, shader_dir="src/shaders"):
        self.logger = logging.getLogger(__name__)
        self.shaders = {}
        self.current_shader = None
        self.shader_dir = shader_dir
        
        try:
            version = glGetString(GL_SHADING_LANGUAGE_VERSION)
            ver_str = version.decode('utf-8') if version else 'unknown'
            self.logger.info(f"GLSL Version: {ver_str}")
            self.glsl_supported = True
        except Exception as e:
            self.logger.warning(f"GLSL not supported: {e}")
            self.glsl_supported = False

    def check_shader_support(self):
        if not self.glsl_supported:
            print("WARNING: GLSL support unavailable.")
            return False
        version = glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')
        print(f"OpenGL Shader Language (GLSL) Version: {version}")
        print("Shader support is available")
        return True

    def validate_shader_displacement(self, name, vertex_count):
        if not self.glsl_supported or name not in self.shaders:
            print(f"WARNING: Shader '{name}' not available for testing.")
            return False
        program = self.shaders[name]
        glUseProgram(program)
        for uni in ("useHeightMap", "heightScale", "heightMap", "diffuseMap", "seaLevel"):
            loc = glGetUniformLocation(program, uni)
            print(f"Uniform '{uni}': location {loc}")
        print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
        print(f"Vendor: {glGetString(GL_VENDOR).decode('utf-8')}")
        print(f"Renderer: {glGetString(GL_RENDERER).decode('utf-8')}")
        print(f"Expected to displace ~{vertex_count} vertices.")
        glUseProgram(0)
        return True

    def load_shader_from_files(self, name):
        if not self.glsl_supported:
            self.logger.error("Cannot load shaders: GLSL unsupported.")
            return False
        vert_path = os.path.join(self.shader_dir, f"{name}.vert")
        frag_path = os.path.join(self.shader_dir, f"{name}.frag")
        try:
            with open(vert_path) as f:
                vert_src = f.read()
            with open(frag_path) as f:
                frag_src = f.read()
        except IOError as e:
            self.logger.error(f"Shader file not found: {e}")
            return False
        try:
            program = compileProgram(
                compileShader(vert_src, GL_VERTEX_SHADER),
                compileShader(frag_src, GL_FRAGMENT_SHADER)
            )
            glUseProgram(program)
            for uniform, unit in (("diffuseMap", 0), ("heightMap", 1)):
                loc = glGetUniformLocation(program, uniform)
                if loc != -1:
                    glUniform1i(loc, unit)
            glUseProgram(0)
            self.shaders[name] = program
            self.logger.info(f"Loaded shader '{name}' from files.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to compile shader '{name}': {e}")
            return False

    def use_shader(self, name):
        if not self.glsl_supported or name not in self.shaders:
            return False
        glUseProgram(self.shaders[name])
        self.current_shader = name
        return True

    def stop_using_shader(self):
        if self.glsl_supported:
            glUseProgram(0)
            self.current_shader = None

    def set_uniform_bool(self, name, value):
        if not self.glsl_supported or not self.current_shader:
            return
        loc = glGetUniformLocation(self.shaders[self.current_shader], name)
        if loc != -1:
            glUniform1i(loc, int(bool(value)))

    def set_uniform_float(self, name, value):
        if not self.glsl_supported or not self.current_shader:
            return
        loc = glGetUniformLocation(self.shaders[self.current_shader], name)
        if loc != -1:
            glUniform1f(loc, float(value))
            
    def set_uniform_vec3(self, name, x, y, z):
        if not self.glsl_supported or not self.current_shader:
            return
        loc = glGetUniformLocation(self.shaders[self.current_shader], name)
        if loc != -1:
            glUniform3f(loc, float(x), float(y), float(z))