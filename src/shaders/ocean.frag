#version 120

uniform sampler2D diffuseMap;
uniform sampler2D heightMap;
uniform bool useHeightMap;
uniform float heightScale;
uniform float seaLevel;
uniform vec3 oceanColor;
uniform vec3 shoreColor;

varying vec2 texCoord;
varying vec3 normal;
varying vec3 position;

void main() {
    float h = texture2D(heightMap, texCoord).r;
    vec4 landColor = texture2D(diffuseMap, texCoord);
    vec3 finalColor;
    float alpha = 1.0;

    if (h < seaLevel) {
        float depth = (seaLevel - h) / seaLevel;
        float shoreFactor = 1.0 - depth;
        
        finalColor = mix(oceanColor, shoreColor, shoreFactor * 0.7);
        
        alpha = mix(0.4, 0.85, shoreFactor);
        
        vec3 viewDir = normalize(-position);
        float fresnel = pow(1.0 - max(0.0, dot(normalize(normal), viewDir)), 3.0);
        alpha = mix(alpha, 1.0, fresnel * 0.5);
    } else {
        if (useHeightMap) {
            float factor = clamp((h - seaLevel) * heightScale * 5.0, 0.0, 1.0);
            landColor.rgb = mix(landColor.rgb * 0.7, landColor.rgb * 1.3, factor);
        }
        
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normalize(normal), lightDir), 0.0);
        finalColor = landColor.rgb * (0.3 + 0.7 * diff);
    }

    gl_FragColor = vec4(finalColor, alpha);
}