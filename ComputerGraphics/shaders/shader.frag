#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 Position_worldspace;
layout(location = 2) in vec3 Normal_cameraspace;
layout(location = 3) in vec3 EyeDirection_cameraspace;
layout(location = 4) in vec3 LightDirection_cameraspace;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D cubeTexture;

layout(binding = 2) uniform Light {
    float intensity;
	vec3 color;
    vec3 position;
} light;

void main() {
    // Light emission properties
	// You probably want to put them as uniforms
	float LightPower = light.intensity;

	// Material properties
	vec3 MaterialDiffuseColor = texture(cubeTexture, fragTexCoord).rgb;
	vec3 MaterialAmbientColor = vec3(0.1,0.1,0.1) * MaterialDiffuseColor;
	vec3 MaterialSpecularColor = vec3(0.3,0.3,0.3);

	// Distance to the light
	float distance = length( light.position - Position_worldspace );

	// Normal of the computed fragment, in camera space
	vec3 n = normalize( Normal_cameraspace );
	// Direction of the light (from the fragment to the light)
	vec3 l = normalize( LightDirection_cameraspace );
	// Cosine of the angle between the normal and the light direction, 
	// clamped above 0
	//  - light is at the vertical of the triangle -> 1
	//  - light is perpendicular to the triangle -> 0
	//  - light is behind the triangle -> 0
	float cosTheta = clamp( dot( n,l ), 0,1 );

	// Eye vector (towards the camera)
	vec3 E = normalize(EyeDirection_cameraspace);
	// Direction in which the triangle reflects the light
	vec3 R = reflect(-l,n);
	// Cosine of the angle between the Eye vector and the Reflect vector,
	// clamped to 0
	//  - Looking into the reflection -> 1
	//  - Looking elsewhere -> < 1
	float cosAlpha = clamp( dot( E,R ), 0,1 );

    outColor = vec4(
		// Ambient : simulates indirect lighting
		MaterialAmbientColor +
		// Diffuse : "color" of the object
		MaterialDiffuseColor * light.color * LightPower * cosTheta / (distance*distance) +
		// Specular : reflective highlight, like a mirror
		MaterialSpecularColor * light.color * LightPower * pow(cosAlpha,5) / (distance*distance),
	1.0);
}