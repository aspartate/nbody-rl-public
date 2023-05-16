using Flux, Random, Statistics

mutable struct Agent
    env::Any
    scaler
    lookback::Int
    epsilon::Float64
    memory_size::Int
    memory::Vector              #   Stores (observation_stack, action, reward, next_observation_stack, done) tuples
    episode_breaks::Vector{Int} #   Stores indices of memory where episodes end
    loss_history::Vector{Float64}
    flattened_observation_dim::Int
    hidden_dim::Int
    output_dim::Int
    layers::Chain
end

function buildAgent(env::Any, scaler, lookback::Int, epsilon::Float64, memory_size::Int)
    # Make a copy of the environment so we can find the length of the flattened observation without modifying the original environment
    env_ = deepcopy(env)
    flattened_observation_dim = length(new_system!(env_))
    
    hidden_dim = 64
    output_dim = length(env.action_space)
    layers = Chain(
        Dense(flattened_observation_dim * lookback => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim),
        softmax
    )

    Agent(
        env,
        scaler,
        lookback,
        epsilon,
        memory_size,
        [], # memory
        [0], # episode_breaks
        [], # loss_history
        flattened_observation_dim,
        hidden_dim,
        output_dim,
        layers
    )
end

function add_lookback(agent::Agent, observation; verbose=false)
    # Expands observation backwards to include n=lookback number of consecutive observations
    # If left bound is reached, pad with duplicate of leftmost observation

    # Initialize stack of observations
    observation_stack = zeros(agent.lookback, agent.flattened_observation_dim)

    # Find index of observation in memory
    if isempty(agent.memory)
        # This case is triggered on agent's first step
        past_observations = [observation]
        observation_index = 1
    else
        past_observations = [t[1] for t in agent.memory]
        observation_found = false
        for i in 1:length(past_observations)
            if all(observation .== past_observations[i])
                observation_index = i
                observation_found = true
                break
            end
        end
        if !observation_found
            # This case is triggered on subsequent steps, but not during batch training
            push!(past_observations, observation)
            observation_index = length(past_observations)
        end
    end
    
    # Get index of previous episode break (max of self.episode_breaks that is less than observation_index)
    prev_episode_end = maximum([i for i in agent.episode_breaks if i < observation_index])

    if verbose
        println("Prev episode end: ", prev_episode_end)
        println("observation index: ", observation_index)
    end

    # Fill in observation_stack until previous episode is reached
    for i in 1:agent.lookback
        if observation_index - i > prev_episode_end
            # println("Observation index: ", observation_index)
            # println("i: ", i)
            # println("prev_episode_end: ", prev_episode_end)
            observation_stack[i, :] = past_observations[observation_index - i]
        else
            observation_stack[i, :] = past_observations[prev_episode_end + 1] # Pad with duplicate of leftmost observation
        end
    end

    if verbose
        println("Shape of observation stack: ", size(observation_stack))
    end

    return observation_stack
end

function (agent::Agent)(observation_stacks)
    # Note that here observation_stacks has shape (batch_size, lookback, flattened_observation_dim)

    # Scale observation_stacks
    # println("Batch with shape $(size(observation_stacks)): ", observation_stacks)
    observation_stacks = [transform(agent.scaler, observation_stacks[batch, :, :]) for batch in 1:size(observation_stacks, 1)]
    # println("Transformed batch with shape $(size(observation_stacks[1])): ", observation_stacks)

    # Flatten each batch of observation_stacks into a Vector of length (flattened_observation_dim * lookback)
    observation_stacks = [vcat(batch...) for batch in observation_stacks]
    # Convert observation_stacks to a matrix with dims (flattened_observation_dim * lookback, batch_size)
    observation_stacks = hcat(observation_stacks...)
    # Convert observation_stacks to Matrix{Float32}
    observation_stacks = Matrix{Float32}(observation_stacks)

    # Return Q values for all actions
    return agent.layers(observation_stacks)
end

function act(agent::Agent, observation; eval_mode=false, step_boost=0)
    # Choose action based on modified epsilon-greedy policy with fixed epsilon
    # With probability 0.5*epsilon choose the timestep above the favored timestep.
    # With probability 0.5*epsilon choose the timestep below the favored timestep.
    # Otherwise choose the favored timestep.
    # Based on: https://github.com/lueckem/quadrature-ML/blob/master/time_stepper_ODE.ipynb

    # Add lookback to observation
    observation_stack = add_lookback(agent, observation)

    # Add batch dimension
    observation_stack = reshape(observation_stack, (1, size(observation_stack, 1), size(observation_stack, 2)))
    
    # Get Q values for all actions
    q_values = agent(observation_stack)
    # Get index of favored action
    favored_idx = argmax(q_values[:, 1])
    # Apply step_boost
    favored_idx = min(max(favored_idx + step_boost, 1), length(agent.env.action_space) - 1)

    # Set epsilon to 0 if in eval mode
    if eval_mode
        agent.epsilon = 0
    end

    # Choose action based on epsilon-greedy policy, with step_boost if provided
    if rand() < agent.epsilon
        if rand() < 0.5
            return min(favored_idx + 1, length(agent.env.action_space) - 1)
        else
            return max(favored_idx - 1, 1)
        end
    else
        return favored_idx
    end
end

function remember!(agent::Agent, observation, action, reward, next_observation, done)
    # Remove oldest transition if memory is full
    if length(agent.memory) >= agent.memory_size
        popfirst!(agent.memory)
        # Update episode_breaks since indices have shifted
        agent.episode_breaks = [max(0, i - 1) for i in agent.episode_breaks]
        # Remove duplicate episode breaks but keep order
        agent.episode_breaks = unique(sort(agent.episode_breaks))
        # println("wARNING: memory is full, oldest transition removed")
        # println("New episode breaks: ", agent.episode_breaks)
        # println("New memory length: ", length(agent.memory))
    end

    # Store transition in memory
    push!(agent.memory, (observation, action, reward, next_observation, done))

    # Update episode_breaks if episode is done
    if done
        push!(agent.episode_breaks, length(agent.memory) - 1)
    end
end

function train!(agent::Agent, batch_size, optimizer, gamma)
    if length(agent.memory) < batch_size
        return
    end

    # Sample a batch of transitions from memory
    batch = rand(agent.memory, batch_size)
    observations, actions, rewards, next_observations, dones = zip(batch...)

    # Convert from NTuple to Vector
    rewards = collect(rewards)
    dones = collect(dones)

    # Expand observations and next_observations to include lookback
    observation_stacks = [add_lookback(agent, observation) for observation in observations]
    next_observation_stacks = [add_lookback(agent, next_observation) for next_observation in next_observations]
    
    # Convert vectors of matrices to arrays of dim (batch_size, lookback, flattened_observation_dim)
    observation_stacks = permutedims(cat(observation_stacks..., dims=3), (3, 1, 2))
    next_observation_stacks = permutedims(cat(next_observation_stacks..., dims=3), (3, 1, 2))

    # Compute target Q values for action taken using Bellman equation
    next_q_values = agent(next_observation_stacks)
    max_next_q_values = vec(maximum(next_q_values, dims=1))
    target_q_values_for_actions = rewards + gamma * max_next_q_values .* (1 .- dones)

    # Get predicted_q_values and update the q values for the actions taken
    target_q_values = agent(observation_stacks)
    for i in 1:length(actions)
        target_q_values[actions[i], i] = target_q_values_for_actions[i]
    end

    # Scale observation_stacks
    observation_stacks = [transform(agent.scaler, observation_stacks[batch, :, :]) for batch in 1:size(observation_stacks, 1)]

    # Flatten each batch of observation_stacks into a Vector of length (flattened_observation_dim * lookback)
    observation_stacks = [vcat(batch...) for batch in observation_stacks]
    observation_stacks = hcat(observation_stacks...)
    observation_stacks = Matrix{Float32}(observation_stacks)

    # Train for one step
    loss, grads = Flux.withgradient(agent.layers) do layers
        Flux.mse(layers(observation_stacks), target_q_values)
    end
    Flux.update!(optimizer, agent.layers, grads[1])
    push!(agent.loss_history, loss)
end
