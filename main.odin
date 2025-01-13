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
WINDOW_WIDTH  :: 640 
WINDOW_HEIGHT :: 480  
VIEW_WIDTH    :: 1.5 * f32(WINDOW_WIDTH)
VIEW_HEIGHT   :: 1.5 * f32(WINDOW_HEIGHT)

// Simulation Cosntants
GRAVITY       :: linalg.Vector2f32{0, -9.8} 	// Gravitational Force
REST_DENS     :: 300.0 				// Resting Density
GAS_CONST     :: 2000.0 			// Constant Gas State
H	      :: 16.0 				// Kernel Radius
HSQ           :: H * H 				// Kernel Radius Squared for optimization
MASS          :: 2.5				// Mass for particles
VISC          :: 200.0				// Viscocity
DT            :: 0.001				// Change in time

// Kernal Constants (Smoothing kernels adapted to 2D based on the SPH paper by Solenthaler)
H_POW_8       :: H * H * H * H * H * H * H * H
H_POW_5       :: H * H * H * H * H
POLY6         :: 4.0 / H_POW_8
SPIKY_GRAD    :: -10.0 / (math.PI * H_POW_5)
VISC_LAP      :: 40.0 / (math.PI * H_POW_5)

// Simulation Paramters
EPS           :: H
BOUND_DAMPING :: -0.5

// Interation Constants
MAX_PARTICLES :: 100000
DAM_PARTICLES :: 500
BLC_PARTICLES :: 250

// Define our Particle
Particle :: struct {
    x, v, f: linalg.Vector2f32,
    rho, p: f32
}

CircleVertex :: struct {
    pos: linalg.Vector2f32
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
    init_sph(MAX_PARTICLES)

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

    glfw.WindowHint(glfw.SAMPLES, 4)
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
    glfw.SetMouseButtonCallback(window, mouse_button_callback)
    gl.load_up_to(3,3, glfw.gl_set_proc_address)
    gl.Enable(gl.MULTISAMPLE)

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

init_sph :: proc(num_particles: int) {
    for y: = EPS; f32(y) < VIEW_HEIGHT - EPS * 2; y += H {
	for x := VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H {
	    if len(particles) < num_particles {
		jitter := f32(rand.float32())
                append(&particles, Particle{linalg.Vector2f32{x + jitter, f32(y)}, linalg.Vector2f32{0,0}, linalg.Vector2f32{0,0}, 0, 0})
	    }
	    else {
		return
	    }
	}
    }
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

mouse_button_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: i32) {
    if action == glfw.PRESS {
        x, y: f64
        glfw.GetCursorPos(window)
        
        // Convert screen coordinates to world coordinates
        mouse_pos := linalg.Vector2f32{f32(x) * VIEW_WIDTH / f32(WINDOW_WIDTH), f32(WINDOW_HEIGHT - y) * VIEW_HEIGHT / f32(WINDOW_HEIGHT)}
        
        force_magnitude := 500.0 // Adjust this value to change how strong the force is
        force_radius := 100.0   // Distance where the force effect drops to zero
        
        for &p in &particles {
            dir := mouse_pos - p.x
            distance := linalg.length(dir)
            if distance < f32(force_radius) {
                // Normalize direction and calculate force
                dir_norm := dir
                
                if button == glfw.MOUSE_BUTTON_LEFT {
                    p.f += dir_norm * f32(force_magnitude) * (1.0 - distance / f32(force_radius))
                } else if button == glfw.MOUSE_BUTTON_RIGHT {
                    p.f -= dir_norm * f32(force_magnitude) * (1.0 - distance / f32(force_radius))
                }
            }
        }
    }
}

render :: proc(window: glfw.WindowHandle) {
      gl.Clear(gl.COLOR_BUFFER_BIT)

    // Create vertices for a small circle
    num_segments := 12 // Number of triangles; more for smoother circles
    circle_vertices := make([]CircleVertex, num_segments * 3)
    
    for p, i in particles {
        radius := H / 2.0 // or adjust to your liking
        for j := 0; j < num_segments; j += 1 {
            theta := f32(j) / f32(num_segments) * 2 * math.PI
            next_theta := f32(j+1) / f32(num_segments) * 2 * math.PI
            
            circle_vertices[j*3+0] = CircleVertex{p.x}
            circle_vertices[j*3+1] = CircleVertex{linalg.Vector2f32{p.x.x + f32(radius) * math.cos(theta), p.x.y + f32(radius) * math.sin(theta)}}
            circle_vertices[j*3+2] = CircleVertex{linalg.Vector2f32{p.x.x + f32(radius) * math.cos(next_theta), p.x.y + f32(radius) * math.sin(next_theta)}}
        }

        // Use this for drawing
        gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
        gl.BufferData(gl.ARRAY_BUFFER, len(circle_vertices) * size_of(CircleVertex), &circle_vertices[0], gl.DYNAMIC_DRAW)

        // Assuming you've set up your shaders for a vec2 position
        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, size_of(CircleVertex), 0)
        gl.EnableVertexAttribArray(0)

        // Draw the triangles for this particle
        gl.DrawArrays(gl.TRIANGLES, 0, i32(len(circle_vertices)))
    }
    
    // Set up our projection matrix
    proj_matrix := linalg.matrix_ortho3d_f32(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1)
    gl.UseProgram(program)

    // Set the projection matrix uniform location
    uniform_location := gl.GetUniformLocation(program, "projection")
    gl.UniformMatrix4fv(uniform_location, 1, gl.FALSE, &proj_matrix[0][0])
}

compute_density_pressure :: proc() {
    for &pi in &particles {
	pi.rho = 0
	for pj in particles {
	    rij := pj.x - pi.x
	    r2 := linalg.length2(rij)
	    if r2 < HSQ {
		pi.rho += MASS * POLY6 * math.pow_f32(HSQ - r2, 3.0) 
	    }
	}

	pi.p = GAS_CONST * (pi.rho - REST_DENS)
    }
}

compute_forces :: proc () {
    for &pi in &particles {
	pi.f = linalg.Vector2f32{0,0}

	for pj in particles {
	    if pi == pj { continue }

	    rij := pj.x - pi.x
	    r := linalg.length(rij)

	    if r < H {
		// Pressure force here
		grad_factor := SPIKY_GRAD * math.pow_f32(H - r, 3.0)
		fpress := -rij * MASS * (pi.p + pj.p) / (2.0 * pj.rho) * grad_factor // check back on this

		// Viscocity force
		fvisc := VISC * MASS * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r)

		pi.f += fpress + fvisc
	    }
	}

	// Gravity force
	pi.f += GRAVITY * MASS / pi.rho
    }
}

integrate :: proc() {
    for &p in &particles {
	// Update velocity with force
	p.v += DT * p.f / p.rho
	p.x += DT * p.v

	// Boundary conditions
	if p.x.x - EPS < 0 {
	    p.v.x *= BOUND_DAMPING
	    p.x.x = EPS
	}

	if p.x.x + EPS > VIEW_WIDTH {
	    p.v.x *= BOUND_DAMPING
	    p.x.x = VIEW_WIDTH - EPS
	}

	if p.x.y - EPS < 0 {
	    p.v.y *= BOUND_DAMPING
	    p.x.y = EPS
	}

	if p.x.y + EPS > VIEW_HEIGHT {
	    p.v.y *= BOUND_DAMPING
	    p.x.y = VIEW_HEIGHT - EPS
	}
    }
}

update :: proc() {
    compute_density_pressure()
    compute_forces()
    integrate()
}
