
using Plots
using LinearAlgebra

function test_lorenz_environment()
    action_space = [0.02, 0.022, 0.025, 0.029, 0.033, 0.039, 0.045, 0.052, 0.060, 0.070]
    tol = 1e-4
    pos_reward_range = (0.1, 2.0)
    max_time = 10.0
    method = "RK4"
    num_episodes = 10

    # Initialized environment
    env = buildLorenzEnvironment(action_space, tol, pos_reward_range, max_time, method)
    
    reward_history = []
    error_history = []
    global_step = 0
    for episode in 1:num_episodes
        # Reset environment
        observation = new_system!(env)
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
        println("Episode $episode | Global Step $global_step | Total Reward $(sum(episode_rewards)) | Avg Reward $(sum(episode_rewards) / length(episode_rewards)) | Avg Error/Tol $(sum(episode_errors) / length(episode_errors) / tol)")
    end

    # Plot reward history
    display(plot(1:num_episodes, reward_history, title="Avg. Episode Rewards", xlabel="Episode", ylabel="Avg. Reward", legend=false))

    # Plot error history
    display(plot(1:global_step, error_history, title="Error History", xlabel="Global Step", ylabel="Error", legend=false))
end

