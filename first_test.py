import glfw
from OpenGL.GL import *
import ctypes
import numpy as np

# same seed for testing purposes
# np.random.seed(42)

def generate_random_spheres_new(num_spheres, min_position, max_position, min_radius, max_radius, min_color, max_color,min_reflect,max_reflect):
    # Generate random positions, radii, and colors for each sphere
    positions = np.random.uniform(min_position, max_position, size=(num_spheres, 3)).astype(np.float32)
    radii = np.random.uniform(min_radius, max_radius, size=num_spheres).astype(np.float32)
    colors = np.random.uniform(min_color, max_color, size=(num_spheres, 3)).astype(np.float32)
    reflectance = np.random.uniform(min_reflect, max_reflect, size=num_spheres).astype(np.float32)

    # Combine positions and radii into position_radius array
    position_radius = np.column_stack((positions, radii))

    # Combine colors with padding into color_padding array
    color_padding = np.column_stack((colors, reflectance))

    combined_array = np.hstack((position_radius, color_padding))


    return combined_array


cs_source = open("./shaders/raytrace.comp","r").read()

width = 800
height = 600


vertex_shader_source = """
#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_source = """
#version 430 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
"""
def process_input(window):
    if (glfw.get_key(window,glfw.KEY_ESCAPE)==glfw.PRESS):
        glfw.set_window_should_close(window,True)

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise Exception("Shader compilation failed: " + glGetShaderInfoLog(shader).decode())

    return shader


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def main():

    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window((width), (height), "hallo", None, None)
    if not window:
        print("unable to create Window")
        glfw.terminate()
        return
    glfw.make_context_current(window)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glViewport(0, 0, width, height)
    glfw.swap_interval(1)
    
    # Create Vertex Array Object (VAO), Vertex Buffer Object (VBO), and Element Buffer Object (EBO)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    # Bind VAO
    glBindVertexArray(VAO)
    

    # Vertex data for a full-screen quad
    vertices = [
        # Positions      # Texture Coordinates
        -1.0, -1.0,      0.0, 0.0,
         1.0, -1.0,      1.0, 0.0,
        -1.0,  1.0,      0.0, 1.0,
         1.0,  1.0,      1.0, 1.0
    ]

    # Bind VBO and set vertex data
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, (GLfloat * len(vertices))(*vertices), GL_STATIC_DRAW)

    # Set vertex attribute pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(2 * sizeof(GLfloat)))
    glEnableVertexAttribArray(1)

    float_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, float_buffer)

    float_array =np.array([0.2,0.4,0.2],dtype=np.float32) 
    print(float_array)
    glBufferData(GL_SHADER_STORAGE_BUFFER, float_array.nbytes, float_array, GL_DYNAMIC_DRAW)
    print(float_array.nbytes)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, float_buffer)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    compute = compile_shader(cs_source,GL_COMPUTE_SHADER)

    vertex_shader = compile_shader(vertex_shader_source,GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source,GL_FRAGMENT_SHADER)


    compute_Programm = glCreateProgram()
    glAttachShader(compute_Programm,compute)
    glLinkProgram(compute_Programm)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    glDeleteShader(compute)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    texture = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D,texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    num_spheres = 100
    min_position = np.array([-5.0, -5.0, -15.0])
    max_position = np.array([5.0, 5.0, -10.0])
    min_color = np.array([0.7,0.7,0.7])
    max_color = np.array([1,1,1])
    min_radius = 0.3
    max_radius = 0.8
    min_reflectance = 0
    max_reflectance = 1


    # mySpheres = generate_random_spheres(num_spheres,min_position,max_position,min_radius,max_radius,min_color,max_color)
    sphere_array = generate_random_spheres_new(num_spheres,min_position,max_position,min_radius,max_radius,min_color,max_color,min_reflectance,max_reflectance)

    # sphere_array = np.array(random_spheres, dtype=[("position_radius", np.float32, 4), ("color_padding", np.float32, 4)])
    # print(sphere_array)

    # print(mySpheres[1].color[0])

    # Buffer-Objekt erstellen
    print(sphere_array)
    sphere_buffer = glGenBuffers(1)

    # Daten auf die GPU übertragen
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphere_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, sphere_array.nbytes, sphere_array, GL_DYNAMIC_DRAW)
    print(sphere_array.nbytes)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sphere_buffer)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    size = glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE)
    print(f"Buffer Size: {size} bytes")
    data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sphere_array.nbytes)
    print(data)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    


    glGetError()
    # buffer_size = len(mySpheres) * ctypes.sizeof(Sphere)
    # print(f"Expected Buffer Size: {buffer_size}")

    
    glUseProgram(shader_program)
    glUniform1i(glGetUniformLocation(shader_program, "textureSampler"), 0)

    t_locations = glGetUniformLocation(compute_Programm,"t")
    
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")


    glClearColor(0.3,0.7,0.3,1.0)

    deltaTime = 0
    lastFrame = 0
    fCounter = 0
    while not(glfw.window_should_close(window)):
        process_input(window)
        currentFrame = glfw.get_time()
        # print(type(currentFrame))
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame
        if fCounter > 60:
            print(f"FPS: {1/deltaTime}")
            fCounter =0
        else:
            fCounter += 1

        glUseProgram(compute_Programm)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUniform1f(t_locations,currentFrame)
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error: {error}")

        glDispatchCompute(int(width/10),int(height/10),1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


        glUseProgram(shader_program)
        glClear(GL_COLOR_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)    

        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()

if __name__ == "__main__":
    main()
