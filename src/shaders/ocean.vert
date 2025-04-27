#version 120

uniform sampler2D heightMap;
uniform bool useHeightMap;
uniform float heightScale;
uniform float seaLevel;

varying vec2 texCoord;
varying vec3 normal;
varying vec3 position;

void main() {
    texCoord = gl_MultiTexCoord0.xy;
    vec4 pos = gl_Vertex;
    normal = gl_Normal;

    position = pos.xyz;
    
    gl_Position = gl_ModelViewProjectionMatrix * pos;
}