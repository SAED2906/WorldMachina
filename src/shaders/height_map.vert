#version 120

uniform sampler2D heightMap;
uniform bool        useHeightMap;
uniform float       heightScale;

varying vec2        texCoord;
varying vec3        normal;

void main() {
    texCoord = gl_MultiTexCoord0.xy;
    normal   = gl_Normal;

    vec4 pos = gl_Vertex;

    if (useHeightMap) {
        float d = texture2D(heightMap, texCoord).r;
        pos.xyz += normal * (d * heightScale);
    }

    gl_Position = gl_ModelViewProjectionMatrix * pos;
}