"""
    Wmatrix(dS::Function, dB::Function, f::Function, lims::Tuple)

Computes the 2x2 weight matrix for signal/background.

# Arguments
- `dS`: Signal PDF as a function.
- `dB`: Background PDF as a function.
- `f`: Total PDF as a function (e.g., mixture of signal and background).
- `lims`: Tuple giving integration limits `(lower, upper)`.

# Returns
- 2×2 matrix `W` where `W[i,j] = ∫ (di(x) * dj(x) / f(x)) dx`

# Example
```julia
dS(x) = pdf(Normal(0,1), x)
dB(x) = pdf(Normal(5,1), x)
f(x) = 0.5*dS(x) + 0.5*dB(x)
W = Wmatrix(dS, dB, f, (-5, 10))
```
"""
function Wmatrix(dS, dB, f, lims::Tuple{<:Real, <:Real})
    comps = [dS, dB]
    W = [
        quadgk(x -> di(x) * dj(x) / f(x), lims...)[1]
        for di in comps, dj in comps
    ]
    return W
end

"""
    Wmatrix(pdfS::MixtureModel, pdfB::MixtureModel, f::Function)

Computes the 2×2 sWeight matrix using signal and background `MixtureModel`s
from `Distributions.jl` and a total PDF function `f(x)`.

# Arguments
- `pdfS`: Signal model, a `MixtureModel` (e.g., `MixtureModel([Normal(...)], [weight])`).
- `pdfB`: Background model, also a `MixtureModel`.
- `f`: Total PDF as a function (e.g., `x -> f_signal * pdf(pdfS, x) + (1 - f_signal) * pdf(pdfB, x)`)

# Returns
- 2×2 matrix `W` where `W[i,j] = ∫ (di(x) * dj(x) / f(x)) dx`, integrated over the union support.

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
f(x) = 0.4 * pdf(pdfS, x) + 0.6 * pdf(pdfB, x)

W = Wmatrix(pdfS, pdfB, f)
```
"""
function Wmatrix(pdfS::MixtureModel, pdfB::MixtureModel, f::Function)
    dS(x) = pdf(pdfS, x)
    dB(x) = pdf(pdfB, x)
    lims = support_union(pdfS, pdfB)
    return Wmatrix(dS, dB, f, lims)
end

"""
    sWeights(pdfS::MixtureModel, pdfB::MixtureModel, fraction_signal::Real)

Computes the sWeight functions for signal and background components
based on mixture modeling.

# Arguments
- `pdfS`: Signal model as a `MixtureModel`.
- `pdfB`: Background model as a `MixtureModel`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).

# Returns
- A pair of functions `(w_signal, w_background)` such that:
    - `w_signal(x)` = sWeight for signal at point `x`
    - `w_background(x)` = sWeight for background at point `x`

# Example
```julia
using Distributions

pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
f_sig = 0.4

wS, wB = sWeights(pdfS, pdfB, f_sig)
println(wS(0.0))  # ≈ 1.0 (near pure signal region)
println(wB(5.0))  # ≈ 1.0 (near pure background region)
```
"""
function sWeights(pdfS::MixtureModel, pdfB::MixtureModel, fraction_signal::Real)
    # Define combined PDF f(x) = fS * pdfS(x) + fB * pdfB(x)
    f(x) = fraction_signal * pdf(pdfS, x) + (1 - fraction_signal) * pdf(pdfB, x)
    # Compute the weight matrix
    W = Wmatrix(pdfS, pdfB, f)
    α = inv(W)
    # Normalize α vectors
    αS = α[:, 1] ./ abs(sum(α[:, 1]))
    αB = α[:, 2] ./ abs(sum(α[:, 2]))
    # Define weight functions
    numerator_s(x) = αS[1] * pdf(pdfS, x) + αS[2] * pdf(pdfB, x)
    numerator_b(x) = αB[1] * pdf(pdfS, x) + αB[2] * pdf(pdfB, x)
    #
    weight_signal(x) = numerator_s(x) / f(x) * fraction_signal
    weight_background(x) = numerator_b(x) / f(x) * (1 - fraction_signal)
    #
    return (weight_signal, weight_background)
end