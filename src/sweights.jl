"""
    sPlot(model::MixtureModel)

Encapsulates the sPlot decomposition for a mixture model.

# Fields
- `model`: The `MixtureModel` (from Distributions.jl).
- `inv_W`: The inverse sPlot weight matrix (covariance of yields).

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1)
model = MixtureModel([pdfS, pdfB], [0.6, 0.4])
sP = sPlot(model)
```
"""
struct sPlot{M, T}
    model::M
    inv_W::Matrix{T}
    #
    function sPlot(model::MixtureModel)
        comps = model.components
        weights = model.prior.p
        f(x) = sum(weights[i] * pdf(comps[i], x) for i in eachindex(comps))
        lims = (minimum([minimum(support(c)) for c in comps]), maximum([maximum(support(c)) for c in comps]))
        ϵ = 1e-12
        W = [quadgk(x -> pdf(ci, x) * pdf(cj, x) / max(f(x), ϵ), lims...)[1]
             for ci in comps, cj in comps]
        inv_W = inv(W)
        return new{typeof(model), eltype(inv_W)}(model, inv_W)
    end
end

"""
    sWeights(sPlot::sPlot, xs::AbstractVector)

Compute the sWeights for each component and each data point.

# Arguments
- `sPlot`: An `sPlot` object containing the mixture model and its inverse weight matrix.
- `xs`: Vector of data points where sWeights are evaluated.

# Returns
- `weights`: Matrix of size (length(xs), n_components), where each column is the sWeights for a component.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1)
model = MixtureModel([pdfS, pdfB], [0.6, 0.4])
sP = sPlot(model)
wS, wB = sWeights(sP, xs) |> eachcol
fS(x) = sWeights(sP, [x])[1,1]
fB(x) = sWeights(sP, [x])[1,2]
```
"""
function sWeights(sP::sPlot, xs::AbstractVector)
    comps = sP.model.components
    weights = sP.model.prior.p
    inv_W = sP.inv_W
    ncomp = length(comps)
    f(x) = sum(weights[i] * pdf(comps[i], x) for i in 1:ncomp)
    α = [inv_W[:, i] ./ abs(sum(inv_W[:, i])) for i in 1:ncomp]
    result = zeros(length(xs), ncomp)
    for i in 1:ncomp
        for (j, x) in enumerate(xs)
            numer = sum(α[i][k] * pdf(comps[k], x) for k in 1:ncomp)
            result[j, i] = numer / f(x) * weights[i]
        end
    end
    return result
end

"""
    sWeights(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution, fraction_signal::Real, xs::AbstractVector)
Computes the sWeight functions for signal and background components
based on individual distributions.
# Arguments
- `pdfS`: Signal model as a `UnivariateDistribution`.
- `pdfB`: Background model as a `UnivariateDistribution`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).
# Returns
- `weights`: Matrix of size (length(xs), n_components), where each column is the sWeights for a component.
# Example
```julia
using Distributions
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
f_sig = 0.4
x = vcat(rand(pdfS, 40), rand(pdfB, 60))
wS, wB = sWeights(pdfS, pdfB, f_sig, x) |> eachcol
fS(x) = sWeights(sP, [x])[1,1]
fB(x) = sWeights(sP, [x])[1,2]
```
"""
function sWeights(
    pdfS::UnivariateDistribution,
    pdfB::UnivariateDistribution,
    fraction_signal::Real,
    xs::AbstractVector,
)
    model = MixtureModel([pdfS, pdfB], [fraction_signal, 1 - fraction_signal])
    sP = sPlot(model)
    return sWeights(sP, xs)
end

"""
    sWeights(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution, n_signal::Real, n_background::Real, xs::AbstractVector)

Compute the sWeight functions for signal and background components using the absolute fitted yields.

# Arguments
- `pdfS`: Signal model as a `UnivariateDistribution`.
- `pdfB`: Background model as a `UnivariateDistribution`.
- `n_signal`: Fitted number of signal events (must be non-negative).
- `n_background`: Fitted number of background events (must be non-negative).

# Returns
- `weights`: Matrix of size (length(xs), n_components), where each column is the sWeights for a component.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
nS, nB = 40, 60
x = vcat(rand(pdfS, 40), rand(pdfB, 60))
wS, wB = sWeights(pdfS, pdfB, nS, nB, x) |> eachcol
fS(x) = sWeights(sP, [x])[1,1]
fB(x) = sWeights(sP, [x])[1,2]
```
"""
function sWeights(
    pdfS::UnivariateDistribution,
    pdfB::UnivariateDistribution,
    n_signal::Real,
    n_background::Real,
    xs::AbstractVector,
)
    N = n_signal + n_background
    f_signal = n_signal / N
    return sWeights(pdfS, pdfB, f_signal, xs)
end

"""
    wMatrix(sP::sPlot)

Get the weight matrix from an sPlot object.

# Arguments
- `sP`: An `sPlot` object.

# Returns
- `W`: The sPlot weight matrix.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1)
model = MixtureModel([pdfS, pdfB], [0.6, 0.4])
sP = sPlot(model)
W = wMatrix(sP)
```
"""
function wMatrix(sP::sPlot)
    return inv(sP.inv_W)
end

"""
    inv_W(sP::sPlot)

Get the inverse weight matrix (covariance matrix) from an sPlot object.

# Arguments
- `sP`: An `sPlot` object.

# Returns
- `inv_W`: The inverse weight matrix.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1)
model = MixtureModel([pdfS, pdfB], [0.6, 0.4])
sP = sPlot(model)
cov = inv_W(sP)
```
"""
function inv_W(sP::sPlot)
    return sP.inv_W
end

"""
    inv_W(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution, fraction_signal::Real)

Return the covariance matrix for a two-component mixture model.

# Arguments
- `pdfS`: Signal model as a `UnivariateDistribution`.
- `pdfB`: Background model as a `UnivariateDistribution`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).

# Returns
- `cov`: Covariance matrix of the yields.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
cov = inv_W(pdfS, pdfB, 0.4)
```
"""
function inv_W(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution, fraction_signal::Real)
    model = MixtureModel([pdfS, pdfB], [fraction_signal, 1 - fraction_signal])
    sP = sPlot(model)
    return inv_W(sP)
end

"""
    check_wMatrix_condition(sP::sPlot)

Warn if the sPlot weight matrix is ill-conditioned, which may indicate numerical instability.

# Arguments
- `sP`: An `sPlot` object.

# Returns
- `condW::Float64`: The condition number of the weight matrix.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1)
model = MixtureModel([pdfS, pdfB], [0.6, 0.4])
sP = sPlot(model)
condW = check_wMatrix_condition(sP)
```
"""
function check_wMatrix_condition(sP::sPlot)
    W = wMatrix(sP)
    condW = cond(W)
    if condW > 1e8
        @warn "Weight matrix is ill-conditioned (condition number = $condW). Results may be unstable."
    end
    return condW
end

"""
    sWeights_vector_with_variance(sP::sPlot, xs)

Compute the sWeights and their variances for data points using the sPlot object.

# Arguments
- `sP`: An `sPlot` object.
- `xs`: Vector of data points.

# Returns
- `(ws_signal, ws_background, var_signal, var_background)`: Tuple of vectors containing the sWeights and their variances for each component.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
model = MixtureModel([pdfS, pdfB], [0.4, 0.6])
sP = sPlot(model)
xs = [-2.0, 0.0, 5.0, 8.0]
ws, wb, vs, vb = sWeights_vector_with_variance(sP, xs)
```
"""
function sWeights_vector_with_variance(sP::sPlot, xs)
    weights = sWeights(sP, xs)
    wS, wB = eachcol(weights)

    # Variance calculation
    V = sP.inv_W
    comps = sP.model.components
    prior_weights = sP.model.prior.p
    f(x) = sum(prior_weights[i] * pdf(comps[i], x) for i in eachindex(comps))

    function variance(component_idx, x)
        ps = [pdf(comp, x) for comp in comps]
        v = 0.0
        for j in eachindex(comps), k in eachindex(comps)
            v += V[component_idx, j] * V[component_idx, k] * ps[j] * ps[k]
        end
        return v / (f(x)^2)
    end

    vS = [variance(1, x) for x in xs]
    vB = [variance(2, x) for x in xs]

    return (wS, wB, vS, vB)
end

"""
    sWeights_vector_with_variance(pdfS, pdfB, fraction_signal, xs)

Compute the sWeights and their variances for data points using individual distributions.

# Arguments
- `pdfS`: Signal model as a `UnivariateDistribution`.
- `pdfB`: Background model as a `UnivariateDistribution`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).
- `xs`: Vector of data points.

# Returns
- `(ws_signal, ws_background, var_signal, var_background)`: Tuple of vectors containing the sWeights and their variances.

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
xs = [-2.0, 0.0, 5.0, 8.0]
ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, 0.4, xs)
```
"""
function sWeights_vector_with_variance(pdfS, pdfB, fraction_signal, xs)
    model = MixtureModel([pdfS, pdfB], [fraction_signal, 1 - fraction_signal])
    sP = sPlot(model)
    return sWeights_vector_with_variance(sP, xs)
end
