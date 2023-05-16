using LinearAlgebra
using Flux
using Random
using ProgressBars
using Statistics

function test_StandardScaler()
    # Generate some example data
    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    println("X: ", X)

    Y = [10.0 20.0 30.0]
    println("Y: ", Y)

    # Create a StandardScaler instance
    scaler = StandardScaler(Vector{Float64}(), Vector{Float64}())

    # Fit to X
    fit!(scaler, X)
    # Print shape of means
    println("Shape of Means: ", size(scaler.means))
    # Print shape of stds
    println("Shape of Stds: ", size(scaler.stds))

    # Transform the data
    X_scaled = transform(scaler, X)
    println("X_scaled: ", X_scaled)
    Y_scaled = transform(scaler, Y)
    println("Y_scaled: ", Y_scaled)

    # Inverse transform the scaled data back to the original scale
    X_inverse_transformed = inverse_transform(scaler, X_scaled)
    println("X_inverse_transformed: ", X_inverse_transformed)
    Y_inverse_transformed = inverse_transform(scaler, Y_scaled)
    println("Y_inverse_transformed: ", Y_inverse_transformed)
end

function test_buildscaler_nbody()
    # Define constants
    action_space = [0.1, 0.2, 0.3]
    tol = 1e-4
    pos_reward_range = (10.0, 100.0)
    max_time = 10.0
    method = "RKDP"
    num_episodes = 10
    n_bodies = 3
    gravity = 1.0

    # Initialize environment and scaler
    env = buildNBodyEnvironment(action_space, tol, pos_reward_range, max_time, method, n_bodies, gravity)
    scaler = build_scaler(env, num_episodes)
    println("Shape of Means: ", size(scaler.means))
    println("Shape of Stds: ", size(scaler.stds))

    # Test scaler
    new_system!(env)
    flattened_trace, error_, reward_, done_ = env_step!(env, rand(1:length(env.action_space)-1))
    flattened_trace_matrix = transpose(hcat([flattened_trace]...))
    transformed_trace = transform(scaler, flattened_trace_matrix)

    println("Original trace: ", flattened_trace_matrix)
    println("transformed_trace: ", transformed_trace)
    println("Inverse transformed trace: ", inverse_transform(scaler, transformed_trace))
end

function test_buildscaler_lorenz()
    # Define constants
    action_space = [0.02, 0.022, 0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    tol = 1e-4
    pos_reward_range = (0.1, 2.0)
    max_time = 10.0
    method = "RKDP"
    num_episodes = 10

    # Initialized environment
    println("Building environment...")
    env = buildLorenzEnvironment(action_space, tol, pos_reward_range, max_time, method)
    println("Environment built.")
    scaler = build_scaler(env, num_episodes)
    println("Shape of Means: ", size(scaler.means))
    println("Shape of Stds: ", size(scaler.stds))

    # Test scaler
    new_system!(env)
    flattened_trace, error_, reward_, done_ = env_step!(env, rand(1:length(env.action_space)-1))
    flattened_trace_matrix = transpose(hcat([flattened_trace]...))
    transformed_trace = transform(scaler, flattened_trace_matrix)

    println("Original trace: ", flattened_trace_matrix)
    println("transformed_trace: ", transformed_trace)
    println("Inverse transformed trace: ", inverse_transform(scaler, transformed_trace))
end