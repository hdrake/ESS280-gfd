using Oceananigans
using Printf
using Oceananigans.Utils
using Oceananigans.Coriolis
using Oceananigans.OutputWriters
using Oceananigans.Units
using Oceananigans.Grids: Center
using Oceananigans.AbstractOperations: @at
using NCDatasets

function EadyModel(;
    f0 = 1e-4,
    N = 1e-2,
    U_shear = 10.0,
    ν = 1e-8,
    κ = 1e-8,

    #domain parameters
    Lx = 4e6,
    Ly = 4e6,
    H = 4e3,
    Nx = 48,
    Ny = 48,
    Nz = 16
)

    Λ = U_shear / H
    #numerics 
    dt = 30.0
    stop_time = 40days
    timestepper = :RungeKutta3

    output_prefix = "/Users/hfdrake/Documents/GFD_Fall25/output/"

    #y should not be periodic should be wall bounded
    #add no slip conditions on the velocties in y and z

    grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0,Lx), y=(0,Ly), z=(-H,0), 
                            topology=(Periodic, Bounded, Bounded))

    # background parameters
    U_background(x,y,z) = Λ * z
    B_background(x,y,z) = - Λ * f0 * y + N^2 * z 

    coriolis = FPlane(f=f0)

    #add boundary conditions
    u_bcs = (FieldBoundaryConditions(
        north = ValueBoundaryCondition(0.0),
        south = ValueBoundaryCondition(0.0),
        top = ValueBoundaryCondition(0.0), 
        bottom = ValueBoundaryCondition(0.0)
    ))

    v_bcs = (FieldBoundaryConditions(
        top = ValueBoundaryCondition(0.0), 
        bottom = ValueBoundaryCondition(0.0)
    ))

    w_bcs = (FieldBoundaryConditions(
        north = ValueBoundaryCondition(0.0),
        south = ValueBoundaryCondition(0.0)
    ))

    #model
    model = NonhydrostaticModel(
        grid = grid, 
        advection = WENO(),
        timestepper = :RungeKutta3,
        coriolis = coriolis,
        tracers =:b, 
        buoyancy = BuoyancyTracer(),
        closure = ScalarDiffusivity(; ν, κ),
        boundary_conditions = (
            u = u_bcs, 
            v = v_bcs, 
            w = w_bcs))

    #initial conditions
    noise(x, y, z) = 1.e-2 * (randn() -0.5)
    U_0(x, y, z) = U_background(x,y,z) + noise(x, y, z) #plus background implied 
    V_0(x, y, z) = 0.0 #noise(x, y, z)
    W_0(x, y, z) = 0.0
    B_0(x, y, z) = B_background(x,y,z) + noise(x, y, z)


    set!(model, u=U_0, v=V_0, w=W_0, b=B_0)

    progress(sim) = @printf("i: % 6d, sim time: % 10s, Δt: % 10s \n",
                                sim.model.clock.iteration,
                                prettytime(sim.model.clock.time),
                                prettytime(sim.model.clock.last_Δt))

    sim = Simulation(model; Δt = 10, stop_time=stop_time)

    #add CFL wizard
    wizard = TimeStepWizard(cfl=0.2, max_change = 1.1, max_Δt=60.0)
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    sim.callbacks[:progress] = Callback(progress, IterationInterval(10)) 

    u, v, w = model.velocities
    b = model.tracers.b
    ζ = ∂x(v) - ∂y(u)

    #Ubg = BackgroundField(U_background, grid)
    #u_prime = Ubg - u

    #EKE = @at (Center, Center, Center) 0.5 * (u_prime^2 + v^2 + w^2)

    global_attributes = Dict(
        "f₀" => f0, 
        "N" => N, 
        "U_shear" => U_shear,
        "ν" => ν, 
        "κ" => κ, 
        "Lx" => Lx,
        "Ly" => Ly, 
        "H" => H, 
        "Nx" => Nx, 
        "Ny" => Ny, 
        "Nz" => Nz
    )

    sim.output_writers[:velocities] = NetCDFWriter(
        model, (; u, v, w),
        filename = joinpath(output_prefix, "velocities40day.nc"),
        schedule = TimeInterval(6hours),
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    sim.output_writers[:buoyancy] = NetCDFWriter(
        model, (; b, ζ),
        filename = joinpath(output_prefix, "buoy_vort40day.nc"),
        schedule = TimeInterval(6hours),
        with_halos = false, 
        global_attributes = global_attributes,
        overwrite_existing = true
    )

    return sim
end

sim = EadyModel(;
    f0 = 1e-4,
    N = 1e-2,
    U_shear = 10,
    ν = 1e-6,
    κ = 1e-6,

    #domain parameters
    Lx = 4e6,
    Ly = 4e6,
    H = 4e3,
    Nx = 48,
    Ny = 48,
    Nz = 16)

# # Check if the model looks okay
# println("Model created successfully")
# println("Grid: ", sim.model.grid)
# println("Initial max |u|: ", maximum(abs, sim.model.velocities.u))
# println("Initial max |b|: ", maximum(abs, sim.model.tracers.b))

# # Try to take just ONE timestep
# try
#     time_step!(sim)
#     println("First timestep succeeded!")
# catch e
#     println("ERROR during first timestep:")
#     println(typeof(e))
#     showerror(stdout, e)
# end

run!(sim)