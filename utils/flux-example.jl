using Flux, Statistics, ProgressMeter

function dummymodule()
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
    truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    model = Chain(
        Dense(2 => 3, tanh),   # activation function inside layer
        BatchNorm(3),
        Dense(3 => 2),
        softmax)       # move model to GPU, if available
    print("Type of model: ", typeof(model))

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = model(noisy)                                 # 2×1000 Matrix{Float32}

    # Prepare one batch of data and target:
    batchsize = 64
    x = noisy[:, 1:batchsize]                           # 2x64 Matrix{Float32}
    target = Flux.onehotbatch(truth, [true, false])     # 2×1000 OneHotMatrix
    y = target[:, 1:batchsize]                          # 2x64 OneHotMatrix

    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

    # Training loop, using the single batch of data 10 times:
    losses = []
    @showprogress for epoch in 1:10
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            println("Type of y_hat: ", typeof(y_hat))
            println("Type of y: ", typeof(y))
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end

    optim # parameters, momenta and output have all changed
    out2 = model(noisy)  # first row is prob. of true, second row p(false)

    mean((out2[1,:] .> 0.5) .== truth)  # accuracy after training
end

dummymodule()
