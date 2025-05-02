#version 120

uniform sampler2D diffuseMap;
uniform sampler2D heightMap;
uniform bool useHeightMap;
uniform float heightScale;

varying vec2 texCoord;
varying vec3 normal;

void main() {
    vec4 base = texture2D(diffuseMap, texCoord);
    if (useHeightMap) {
        float h = texture2D(heightMap, texCoord).r;
        base.rgb = mix(base.rgb * 0.7, base.rgb * 1.3, h * heightScale);
    }
    // vec3 L = normalize(vec3(1.0, 1.0, 1.0));
    // float d = max(dot(normalize(normal), L), 0.0);
    // vec3 diff = d * base.rgb;
    // vec3 amb  = base.rgb * 0.3;
    // gl_FragColor = vec4(amb + diff, base.a);
    gl_FragColor = vec4(base.rgb, base.a);
}