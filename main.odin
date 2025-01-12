package sim

import    "core:fmt"
import    "core:math"
import    "core:math/linalg"
import    "core:math/rand"
import    "core:os"
import    "core:time"
import    "vendor:glfw"
import gl "vendor:OpenGL"


// Render Constants
WINDOW_WIDTH  :: 800
WINDOW_HEIGHT :: 600
VIEW_WIDTH    :: 1.5 * f32(WINDOW_WIDTH)
VIEW_HEIGHT   :: 1.5 * f32(WINDOW_HEIGHT)

// Simulation Cosntants
GAVITY        :: linalg.Vector2f32{0, -9.8} 	// Gravitational Force
REST_DENS     :: 300.0 				// Resting Density
GAS_CONST     :: 2000.0 			// Constant Gas State
H	      :: 16.0 				// Kernel Radius
HSQ           :: H * H 				// Kernel Radius Squared for optimization
MASS          :: 2.5				// Mass for particles
VISC          :: 200.0				// Viscocity
DT            :: 0.0007				// Change in time

// Kernal Constants (Smoothing kernels adapted to 2D based on the SPH paper by Solenthaler)
H_POW_8       :: H * H * H * H * H * H * H * H
H_POW_5       :: H * H * H * H * H
POLY6         :: 4.0 / H_POW_8
SPIKY_GRAD    :: -10.0 / H_POW_5
VISC_LAP      :: 40.0 / H_POW_5

// Simulation Paramters
EPS           :: H
BOUND_DAMPING :: -0.5

// Interation Constants
MAX_PARTICLES :: 2500
DAM_PARTICLES :: 500
BLC_PARTICLES :: 250

// Define our Particle
Particle :: struct {
    x, v, f: linalg.Vector2f32,
    rho, p: f32
}

// Shader IDs (gets set when shaders are compiled)
program: u32
vao: u32
vbo: u32

// Global State
particles: [dynamic]Particle


main :: proc () {
    window := init_window()
    defer glfw.Terminate()

    init_shaders()
    init_particles(MAX_PARTICLES)

    gl.ClearColor(0.9, 0.9, 0.9, 1)

    for !glfw.WindowShouldClose(window) {
	gl.Clear(gl.COLOR_BUFFER_BIT)
	gl.Enable(gl.POINT_SMOOTH)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.PointSize(H / 2.0)
	gl.Viewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
	render(window)
	glfw.SwapBuffers(window)
	glfw.PollEvents()

        update()	
    }
}

// For processing our shader files
read_file :: proc(file_name: string) -> string {
    data, ok := os.read_entire_file_from_filename(file_name)
    if !ok {
	fmt.println("Failed to read file: {}", file_name)
	os.exit(1)
    }

    return string(data)
}

init_window :: proc () -> glfw.WindowHandle {
    if !glfw.Init() {
	fmt.println("Failed to initialize GLFW")
	os.exit(1)
    }

    glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)

    window := glfw.CreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "2D SPH Fluid Sim", nil, nil)

    if window == nil {
	fmt.println("Failed to create a window")
	glfw.Terminate()
	os.exit(1)
    }

    glfw.MakeContextCurrent(window)
    gl.load_up_to(3,3, glfw.gl_set_proc_address)

    return window
    
}

init_shaders :: proc() {
    vertex_shader_source := read_file("vertex_shader.glsl")
    fragment_shader_source := read_file("fragment_shader.glsl")
    
    vertex_shader := compile_shader(vertex_shader_source, gl.VERTEX_SHADER)
    fragment_shader := compile_shader(fragment_shader_source, gl.FRAGMENT_SHADER)
    program = link_program(vertex_shader, fragment_shader)

    gl.GenVertexArrays(1, &vao)
    gl.BindVertexArray(vao)

    gl.GenBuffers(1, &vbo)
    gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

    gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, size_of(linalg.Vector2f32), 0)
    gl.EnableVertexAttribArray(0)

}

init_particles :: proc(count: int) {
    for y := EPS; f32(y) < VIEW_HEIGHT - EPS * 2; y += H {
        for x := VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H {
            if len(particles) < count && count <= MAX_PARTICLES {
                jitter := f32(rand.float32())
                append(&particles, Particle{linalg.Vector2f32{x + jitter, f32(y)}, linalg.Vector2f32{0,0}, linalg.Vector2f32{0,0}, 0, 0})
            } else {
		fmt.println("Too many particles!")
                os.exit(1)
            }
        }
    }
}

init_sph :: proc() {

}

compile_shader :: proc(source: string, shader_type: u32) -> u32 {
    shader := gl.CreateShader(shader_type)
    source_str := cstring(raw_data(source))
    gl.ShaderSource(shader, 1, &source_str, nil)
    gl.CompileShader(shader)

    var := i32(0)
    gl.GetShaderiv(shader, gl.COMPILE_STATUS, &var)
    if var == 0 {
	fmt.println("Shader compilation failed!")
    }

    return shader
}

link_program :: proc(vertex_shader: u32, fragment_shader: u32) -> u32 {
    program := gl.CreateProgram()
    gl.AttachShader(program, vertex_shader)
    gl.AttachShader(program, fragment_shader)
    gl.LinkProgram(program)

    var := i32(0)
    gl.GetProgramiv(program, gl.LINK_STATUS, &var)
    if var == 0 {
	fmt.println("Program linking failed")
    }

    return program
}

render :: proc(window: glfw.WindowHandle) {
    gl.Clear(gl.COLOR_BUFFER_BIT)

    //Update vertex buffer output
    data := make([]f32, len(particles) * 2)
    for p, i in particles {
	data[i * 2] = p.x.x
	data[i * 2 + 1] = p.x.y
    }

    gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.BufferData(gl.ARRAY_BUFFER, len(data) * size_of(f32), &data[0], gl.DYNAMIC_DRAW)

    // Set up our projection matrix
    proj_matrix := linalg.matrix_ortho3d_f32(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1)
    gl.UseProgram(program)

    // Set the projection matrix uniform location
    uniform_location := gl.GetUniformLocation(program, "projection")
    gl.UniformMatrix4fv(uniform_location, 1, gl.FALSE, &proj_matrix[0][0])

    // Draw a point for each particle
    gl.BindVertexArray(vao)
    gl.DrawArrays(gl.POINTS, 0, i32(len(particles)))
}

compute_density_pressure :: proc() {

}

compute_forces :: proc () {

}

integrate :: proc() {

}

update :: proc() {
    compute_density_pressure()
    compute_forces()
    integrate()
}
