
using Plots
using LinearAlgebra

function test_nbody_environment()
    N_EPISODES = 3
    MAX_TIME = 1e7
    METHOD = "RKDP"

    ACTION_SPACE = map(x -> x * 2000, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    TOL = 1e-4
    POS_REWARD_RANGE = (0.1, 2.0)

    N_BODIES = 4
    GRAVITY = 6.67430e-11
    INITIAL_STATE = [
        Body(1.989e30, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),              # Sun
        Body(0.33011e24, [57.909e9, 0.0, 0.0], [0.0, 47.36e3, 0.0], [0.0, 0.0, 0.0]),    # Mercury
        Body(4.8675e24, [108.208e9, 0.0, 0.0], [0.0, 35.02e3, 0.0], [0.0, 0.0, 0.0]),   # Venus
        Body(5.972e24, [149.6e9, 0.0, 0.0], [0.0, 29.78e3, 0.0], [0.0, 0.0, 0.0]),      # Earth
    ]

    # Initialized environment
    println("Running simulation for $N_EPISODES episodes with TOL = $TOL, MAX_TIME = $MAX_TIME, N_BODIES = $N_BODIES")
    env = buildNBodyEnvironment(ACTION_SPACE, TOL, POS_REWARD_RANGE, MAX_TIME, METHOD, N_BODIES, GRAVITY)
    
    reward_history = []
    error_history = []
    global_step = 0
    for episode in 1:N_EPISODES
        # Mutate each of the bodies in INITIAL_STATE using the mutate() function
        NEW_INITIAL_STATE = deepcopy(INITIAL_STATE)
        for i in 1:length(NEW_INITIAL_STATE)
            NEW_INITIAL_STATE[i] = mutate(NEW_INITIAL_STATE[i], 0.1)
        end
        # Reset environment
        observation = new_system!(env, NEW_INITIAL_STATE)
        episode_rewards = []
        episode_errors = []
        
        done = false
        while !done
            # Choose a random action
            action = rand(1:length(env.action_space))

            # Take action
            next_observation, error, reward, done = env_step!(env, action)
            push!(episode_rewards, reward)
            push!(episode_errors, error)
            push!(error_history, error)

            # Update global step
            global_step += 1
        end

        # Update reward history with average reward for episode
        push!(reward_history, sum(episode_rewards) / length(episode_rewards))

        # Print progress
        println("Episode $episode | Global Step $global_step | Total Reward $(sum(episode_rewards)) | Avg Reward $(sum(episode_rewards) / length(episode_rewards)) | Avg Error/Tol $(sum(episode_errors) / length(episode_errors)/TOL)")
    end
    # Plot reward history
    display(plot(1:N_EPISODES, reward_history, title="Avg. Episode Rewards", xlabel="Episode", ylabel="Avg. Reward", legend=false))

    # Plot error history
    plot(1:global_step, error_history, title="Error History", xlabel="Global Step", ylabel="Error", legend=false)
    display(hline!([TOL], linecolor=:red, linestyle=:dash))

    # Plot predicted trajectory of each system over time
    println("Number of systems in env: ", length(env.systems))
    for i in 1:length(env.systems)
        plot_trajectory(env.systems[i])
    end
end

