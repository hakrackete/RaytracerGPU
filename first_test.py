import glfw
from OpenGL.GL import *
import ctypes
import numpy as np
import moderngl as mgl
import glm
from math import * 

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
fov = 60

mouse_x = 0
mouse_y = 0
last_x = 0
last_y = 0

deltaTime = 0


cameraPos   = glm.vec3(0.0, 0.0,  0.0);
cameraFront = glm.vec3(0.0, 0.0, -1.0);
cameraUp    = glm.vec3(0.0, 1.0,  0.0);


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

def calc_direction(x,y):
    sensitivity = 0.1
    yaw = -x* sensitivity - 90
    pitch = -y*sensitivity
    pitch = max(-89,min(pitch,89))


    directionvec = glm.vec3()
    directionvec.x = cos(glm.radians(yaw)) * cos(glm.radians(pitch))
    directionvec.y = sin(glm.radians(pitch))
    directionvec.z = sin(glm.radians(yaw)) * cos(glm.radians(pitch))
    directionvec = glm.normalize(directionvec)
    return directionvec



def process_input(window):
    global fov,cameraPos, deltaTime

    cameraspeed = 10 * deltaTime
    fovspeed = 40 * deltaTime
    if (glfw.get_key(window,glfw.KEY_ESCAPE)==glfw.PRESS):
        glfw.set_window_should_close(window,True)
    if (glfw.get_key(window,glfw.KEY_UP)==glfw.PRESS):
        fov += fovspeed 
    if (glfw.get_key(window,glfw.KEY_DOWN)==glfw.PRESS):
        fov -= fovspeed
    if (glfw.get_key(window,glfw.KEY_W)==glfw.PRESS):
        cameraPos.z += cameraspeed 
    if (glfw.get_key(window,glfw.KEY_S)==glfw.PRESS):
        cameraPos.z -= cameraspeed
    if (glfw.get_key(window,glfw.KEY_A)==glfw.PRESS):
        cameraPos.x += cameraspeed 
    if (glfw.get_key(window,glfw.KEY_D)==glfw.PRESS):
        cameraPos.x -= cameraspeed


def mouse_callback(window,xpos,ypos):
    global width,height,mouse_x,mouse_y
    mouse_x = xpos
    mouse_y = height-ypos

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise Exception("Shader compilation failed: " + glGetShaderInfoLog(shader).decode())

    return shader


def framebuffer_size_callback(window, w, h):
    global width, height
    # jittering and artifacts may happen on resizing the image
    # caused by the size of the workgroups beeing a multiple of 10, while framebuffer is not

    # w = int(w / 10) * 10 
    # h = int(h / 10) * 10 
    width = w
    height = h
    glViewport(0, 0, w, h)


def main():
    global deltaTime, cameraFront

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
    glfw.set_cursor_pos_callback(window,mouse_callback)
    glViewport(0, 0, width, height)

    # wenn das fentser im Fokus ist, wird der cursor deaktiviert und in die Mitte des screens gesetzt
    glfw.set_input_mode(window,glfw.CURSOR,glfw.CURSOR_DISABLED)

    # 0 = unlimited FPS, 1 = 60 FPS
    glfw.swap_interval(1)
    
    # # test ob modernGL zusammen mit klassischem OpenGL funktioniert
    # ctx = mgl.create_context()
    # print("Default framebuffer is:", ctx.program()['t'])

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
    min_color = np.array([0.3,0.0,0.0])
    max_color = np.array([1,1,1])
    min_radius = 0.3
    max_radius = 0.8
    min_reflectance = 0
    max_reflectance = 1


    sphere_array = generate_random_spheres_new(num_spheres,min_position,max_position,min_radius,max_radius,min_color,max_color,min_reflectance,max_reflectance)
    sphere_buffer = glGenBuffers(1)

    # Daten auf die GPU übertragen
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphere_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, sphere_array.nbytes, sphere_array, GL_DYNAMIC_DRAW)
    # print(sphere_array.nbytes)
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
    fov_location = glGetUniformLocation(compute_Programm,"myFov")
    view_location = glGetUniformLocation(compute_Programm,"camToWorld")
    x_loc = glGetUniformLocation(compute_Programm,"mouse_x")
    y_loc = glGetUniformLocation(compute_Programm,"mouse_y")


    
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")


    glClearColor(0.3,0.7,0.3,1.0)

    lastFrame = 0
    fCounter = 0
    fps = 0
    view = np.array(glm.lookAt(cameraPos,cameraPos + cameraFront,cameraUp))
    while not(glfw.window_should_close(window)):
        # print(cameraPos)
        offset_x = last_x - mouse_x
        offset_y = last_y - mouse_y
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None);
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        process_input(window)

        currentFrame = glfw.get_time()

        direction = calc_direction(mouse_x,mouse_y)
        cameraFront = direction
        view = np.array(glm.lookAt(cameraPos,cameraPos + cameraFront,cameraUp))
        # print(type(currentFrame))
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame
        
        if fCounter  > fps:
            print(f"FPS: {1/deltaTime}")
            fps = 1/deltaTime
            print(f"time passed:{currentFrame}")
            print(f"x: {mouse_x} y: {mouse_y}")
            fCounter =0
        else:
            fCounter += 1


        glUseProgram(compute_Programm)
        glUniform1f(t_locations,currentFrame)
        glUniform1f(fov_location,fov)
        glUniform1f(x_loc,mouse_x)
        glUniform1f(y_loc,mouse_y)
        
        
        # dunno why, but when i invert it, it works
        glUniformMatrix4fv(view_location,1,GL_TRUE,view)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
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
