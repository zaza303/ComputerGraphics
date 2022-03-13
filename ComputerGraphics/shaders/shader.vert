#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormals;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 Position_worldspace;
layout(location = 2) out vec3 Normal_cameraspace;
layout(location = 3) out vec3 EyeDirection_cameraspace;
layout(location = 4) out vec3 LightDirection_cameraspace;


layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) uniform Light {
    float intensity;
    vec3 color;
    vec3 position;
} light;

void main() {
    vec3 lightPosition = vec3(5.0, 5.0, 5.0);
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // Position of the vertex, in worldspace : M * position
    Position_worldspace = (ubo.model * vec4(inPosition, 1.0)).xyz;
    // Vector that goes from the vertex to the camera, in camera space.
	// In camera space, the camera is at the origin (0,0,0).
	vec3 vertexPosition_cameraspace = ( ubo.view * ubo.model * vec4(inPosition, 1.0)).xyz;
	EyeDirection_cameraspace = vec3(0,0,0) - vertexPosition_cameraspace;
    // Vector that goes from the vertex to the light, in camera space. 'model' is ommited because it's identity.
	vec3 LightPosition_cameraspace = ( ubo.view * vec4(lightPosition, 1.0)).xyz;
	LightDirection_cameraspace = LightPosition_cameraspace + EyeDirection_cameraspace;
    // Normal of the the vertex, in camera space
	Normal_cameraspace = ( ubo.view * ubo.model * vec4(inNormals, 0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.


    fragTexCoord = inTexCoord;
}