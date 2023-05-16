using Random
using Test

# Define a function to print a Body object
function print_body(body::Body)
    println("Mass: ", body.mass)
    println("Position: ", body.position)
    println("Velocity: ", body.velocity)
    println("Acceleration: ", body.acceleration)
end

# Generate random bodies
function generate_random_bodies(num_bodies::Int, seed::Int=42)
    # Random.seed!(seed)
    bodies = Body[]
    for _ in 1:num_bodies
        mass = 1 + 9 * rand()
        position = rand(3) * 100 .- 5
        velocity = zeros(3)
        acceleration = zeros(3)
        push!(bodies, Body(mass, position, velocity, acceleration))
    end
    return bodies
end

function test_nbody_system(num_steps, dt)

    # Test the mutate function
    println("Testing mutate function...")
    # Create a Body object
    body = Body(10.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
    println("Original Body:", body)
    println()

    # Mutate the Body object
    mutated_body = mutate(body, 0.1)
    println("Mutated Body:", mutated_body)

    # Test the NBodySystem implementation
    println("Testing NBodySystem implementation...")
    
    GRAVITY = 6.67430e-11
    INITIAL_STATE = [
        Body(1.989e30, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),              # Sun
        Body(0.33011e24, [57.909e9, 0.0, 0.0], [0.0, 47.36e3, 0.0], [0.0, 0.0, 0.0]),    # Mercury
        Body(4.8675e24, [108.208e9, 0.0, 0.0], [0.0, 35.02e3, 0.0], [0.0, 0.0, 0.0]),   # Venus
        Body(5.972e24, [149.6e9, 0.0, 0.0], [0.0, 29.78e3, 0.0], [0.0, 0.0, 0.0]),      # Earth
        Body(7.34767309e22, [149.6e9 + 384400000.0, 0.0, 0.0], [0.0, 29.78e3 + 1022.0, 0.0], [0.0, 0.0, 0.0]), # Moon
        Body(0.64171e24, [227.9e9, 0.0, 0.0], [0.0, 24.13e3, 0.0], [0.0, 0.0, 0.0]),    # Mars
        # Body(1898.13e24, [778.57e9, 0.0, 0.0], [0.0, 13.07e3, 0.0], [0.0, 0.0, 0.0]),   # Jupiter
        # Body(568.34e24, [1433.5e9, 0.0, 0.0], [0.0, 9.69e3, 0.0], [0.0, 0.0, 0.0]),     # Saturn
        # Body(86.813e24, [2872.5e9, 0.0, 0.0], [0.0, 6.81e3, 0.0], [0.0, 0.0, 0.0]),     # Uranus
        # Body(102.413e24, [4495.1e9, 0.0, 0.0], [0.0, 5.43e3, 0.0], [0.0, 0.0, 0.0])    # Neptune
    ]
    METHOD = "RKDP"
    system = buildNBodySystem(INITIAL_STATE, GRAVITY, METHOD)

    # Perform steps and compute ground truth
    for _ in 1:num_steps
        system_step!(system, dt)
        system_gt!(system)
    end

    # Print results
    println("True final state:")
    println(system.true_state[end])
    println("Estimated final state:")
    println(system.estimated_state[end])

    # Plot trajectory
    plot_trajectory(system)

    # Plot energy and relative error
    plotEnergy(system)
    plotRelativeEnergyError(system)
end

# function test_center_initial_state()
#     # Create a test state with three bodies
#     body1 = Body(1.0, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
#     body2 = Body(2.0, [4.0, 5.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
#     body3 = Body(3.0, [7.0, 8.0, 9.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])

#     initial_state = [body1, body2, body3]

#     # Call center_initial_state function
#     centered_state = center_initial_state(initial_state)

#     # Test the length of the returned array
#     @test length(centered_state) == 3

#     # Test center of mass position and velocity
#     center_of_mass = [0.0, 0.0, 0.0]
#     center_of_mass_vel = [0.0, 0.0, 0.0]
#     total_mass = 0.0
#     for body in centered_state
#         center_of_mass .+= body.mass .* body.position
#         center_of_mass_vel .+= body.mass .* body.velocity
#         total_mass .+= body.mass
#     end
#     center_of_mass ./= total_mass
#     center_of_mass_vel ./= total_mass

#     @test center_of_mass ≈ [0.0, 0.0, 0.0] atol = 1e-6
#     @test center_of_mass_vel ≈ [0.0, 0.0, 0.0] atol = 1e-6

#     # Test if body properties are preserved, except for position and velocity
#     for (i, body) in enumerate(centered_state)
#         @test body.mass == initial_state[i].mass
#         @test body.acceleration == initial_state[i].acceleration
#     end
# end