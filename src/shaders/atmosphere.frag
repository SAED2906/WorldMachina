#version 120

varying vec3 vNormal;

uniform vec3 atmosphereColor;

void main() {
    float intensity = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 4.0);
    gl_FragColor = vec4(atmosphereColor, intensity * 0.5);
}
