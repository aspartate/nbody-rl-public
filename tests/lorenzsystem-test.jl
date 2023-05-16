using Random

function test_lorenz_system()

    # Test the LorenzSystem implementation
    METHOD = "RKDP"
    INITIAL_STATE = [10.0, 10.0, 10.0]
    system = buildLorenzSystem(INITIAL_STATE, METHOD)

    # Perform steps and compute ground truth
    num_steps = 100
    dt = 0.02
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
end