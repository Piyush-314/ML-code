# Neural network:
    Make a prediction
    Measure error (loss)
    Use calculus to adjust weights to reduce error

    Neuron:
        z=wx+b
        y^​=f(z)

    𝑤 (w = weight)
    𝑏(b = bias)
    𝑓(f = activation function)
    composition of functions: x→z→y^


    Loss fxn:
        Mean Squared Error: L=(y^​−y)^2
        chain: (w → z → y^ ​→ L)


Backpropogation:
    Neural networks are nested functions: (Loss → Output → Activation → Weights)
    Applying the chain rule backward through the network
    y=f(g(x))
    dy/dx = dy/dg . dg/dx

    We want: min L (minimise the loss)

    We want: ∂𝐿/∂𝑤
    ∂𝐿/∂𝑤 = (∂𝐿/∂y^) * (∂y^/∂z) * (∂z/∂w)


    # Step 1: Loss derivative
        L = (y^​−y)^2
        ∂𝐿/∂y^ = 2(y^​−y)

    # Step 2: Activation derivative
        y^ = 𝜎(z)
        ∂y^/∂z = σ(z)(1−σ(z))

    # Step 3: Linear derivative
        z = wx + b
        ∂z/∂w = x

    # Final
        ∂𝐿/∂𝑤 = 2(y^​−y) * σ(z)(1−σ(z)) * x


# Gradient Descent
    An algorithm to minimize a function (downwar descent    )
    θ=θ−α∇L

    θ = parameters (weights)
    α = learning rate
    ∇L = gradient (vector of slope)  ∇L=[∂w/∂L ​∂b/∂L​​]
	​
    w = w − α.(∂w/∂L)​
    b = b − α.(∂b/∂L)​


#
| Function | Formula        | Derivative         |
| -------- | -------------- | ------------------ |
| Sigmoid  | (1/(1+e^{-x})) | (σ(x)(1-σ(x)))     |
| ReLU     | (\max(0,x))    | 1 if (x>0), else 0 |
| Tanh     | (\tanh(x))     | (1-\tanh^2(x))     |

