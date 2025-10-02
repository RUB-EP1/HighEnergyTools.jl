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

"""
    sWeights_general(yields::AbstractVector, shape_values::AbstractMatrix)

Compute sWeights for a general mixture model with arbitrary number of components.
This is the core generalized implementation that works for both 1D and multi-dimensional cases.

# Arguments
- `yields`: Vector of fitted component yields [N₁, N₂, ..., Nₖ]
- `shape_values`: Matrix where shape_values[i,j] = Pⱼ(xᵢ) is the j-th component's 
  shape function value (without yield normalization) at the i-th data point

# Returns
- `(weights, covariance)`: Tuple containing:
  - `weights`: Matrix of size (n_events, n_components) with sWeights
  - `covariance`: Covariance matrix of the yields (V matrix from Pivk & Le Diberder)

# Algorithm
Implements the Pivk & Le Diberder method (NIM A555 (2005) 356):
- V⁻¹ᵢⱼ = Σₙ Pᵢ(xₙ)Pⱼ(xₙ)/F(xₙ)² where F(x) = Σₖ NₖPₖ(x)
- wᵢ(n) = Σⱼ Vᵢⱼ Pⱼ(xₙ)/F(xₙ)
- Ensures Σₙ wᵢ(n) = Nᵢ (closure condition)

# Example
```julia
# For a 2-component mixture with 100 events
yields = [60.0, 40.0]  # fitted yields
shape_values = rand(100, 2)  # shape values at each event
weights, cov = sWeights_general(yields, shape_values)
```
"""
function sWeights_general(yields::AbstractVector, shape_values::AbstractMatrix)
    n_events, n_components = size(shape_values)
    
    if length(yields) != n_components
        throw(ArgumentError("Number of yields ($(length(yields))) must match number of components ($n_components)"))
    end
    
    # Build inverse covariance matrix V⁻¹
    V_inv = zeros(Float64, n_components, n_components)
    
    @inbounds for event_idx in 1:n_events
        # Calculate total model value F(x) = Σₖ NₖPₖ(x)
        F_total = 0.0
        for k in 1:n_components
            F_total += yields[k] * shape_values[event_idx, k]
        end
        
        if F_total <= 1e-12
            continue  # Skip events with negligible probability
        end
        
        inv_F_squared = 1.0 / (F_total * F_total)
        
        # Accumulate V⁻¹ᵢⱼ = Σₙ Pᵢ(xₙ)Pⱼ(xₙ)/F(xₙ)²
        for i in 1:n_components
            Pi = shape_values[event_idx, i]
            for j in i:n_components
                val = Pi * shape_values[event_idx, j] * inv_F_squared
                V_inv[i, j] += val
                if j != i
                    V_inv[j, i] += val  # Symmetrize
                end
            end
        end
    end
    
    # Invert to get covariance matrix V
    V = try
        inv(V_inv)
    catch err
        @warn "Covariance matrix inversion failed; using pseudo-inverse" exception=err
        pinv(V_inv)
    end
    
    # Ensure numerical symmetry
    V .= (V .+ V') ./ 2
    
    # Compute sWeights: wᵢ(n) = Σⱼ Vᵢⱼ Pⱼ(xₙ)/F(xₙ)
    weights = zeros(Float64, n_events, n_components)
    
    @inbounds for event_idx in 1:n_events
        # Calculate total model value F(x)
        F_total = 0.0
        for k in 1:n_components
            F_total += yields[k] * shape_values[event_idx, k]
        end
        
        if F_total <= 1e-12
            continue  # Leave weights as zero for negligible events
        end
        
        inv_F = 1.0 / F_total
        
        # Compute weights: wᵢ = (Σⱼ Vᵢⱼ Pⱼ) / F
        for i in 1:n_components
            weight_sum = 0.0
            for j in 1:n_components
                weight_sum += V[i, j] * shape_values[event_idx, j]
            end
            weights[event_idx, i] = weight_sum * inv_F
        end
    end
    
    return (weights, V)
end

"""
    sWeights_multidimensional(yields::AbstractVector, shape_functions::AbstractVector{<:Function}, data_points::AbstractMatrix)

Compute sWeights for multi-dimensional data using component shape functions.

# Arguments
- `yields`: Vector of fitted component yields [N₁, N₂, ..., Nₖ]
- `shape_functions`: Vector of functions where shape_functions[i](x) evaluates the i-th component's
  normalized shape function at point x (x can be a vector for multi-dimensional case)
- `data_points`: Matrix where each row is a data point (n_events × n_dimensions)

# Returns
- `(weights, covariance)`: Tuple containing sWeights matrix and covariance matrix

# Example
```julia
# 2D phi-phi analysis example
phi_signal(m1, m2) = convoluted_phi_signal_pdf(m1, ...) * convoluted_phi_signal_pdf(m2, ...)
phi_bg(m1, m2) = convoluted_phi_signal_pdf(m1, ...) * kk_background_pdf(m2, ...)
# ... define other components

shape_funcs = [
    (x) -> phi_signal(x[1], x[2]),
    (x) -> phi_bg(x[1], x[2]),
    # ... other components
]

yields = [N_phiphi, N_phi1kk2, N_kk1phi2, N_kkkk]
data = [m_phi1 m_phi2]  # n_events × 2 matrix

weights, cov = sWeights_multidimensional(yields, shape_funcs, data)
```
"""
function sWeights_multidimensional(yields::AbstractVector, shape_functions::AbstractVector{<:Function}, data_points::AbstractMatrix)
    n_events, n_dims = size(data_points)
    n_components = length(shape_functions)
    
    if length(yields) != n_components
        throw(ArgumentError("Number of yields ($(length(yields))) must match number of shape functions ($n_components)"))
    end
    
    # Evaluate shape functions at all data points
    shape_values = zeros(Float64, n_events, n_components)
    
    for event_idx in 1:n_events
        data_point = data_points[event_idx, :]
        for comp_idx in 1:n_components
            shape_values[event_idx, comp_idx] = shape_functions[comp_idx](data_point)
        end
    end
    
    # Use the general implementation
    return sWeights_general(yields, shape_values)
end

"""
    check_sweights_closure(weights::AbstractMatrix, yields::AbstractVector; rtol::Float64=1e-3)

Check the closure property of sWeights: Σₙ wᵢ(n) ≈ Nᵢ for each component i.

# Arguments
- `weights`: sWeights matrix (n_events × n_components)
- `yields`: Vector of fitted yields
- `rtol`: Relative tolerance for the closure check

# Returns
- `closure_test`: NamedTuple with fields:
  - `passed`: Boolean indicating if all components pass the closure test
  - `relative_errors`: Vector of relative errors for each component
  - `sums`: Vector of actual sums Σₙ wᵢ(n) for each component

# Example
```julia
weights, _ = sWeights_general(yields, shape_values)
closure = check_sweights_closure(weights, yields)
if !closure.passed
    @warn "sWeights closure test failed" closure.relative_errors
end
```
"""
function check_sweights_closure(weights::AbstractMatrix, yields::AbstractVector; rtol::Float64=1e-3)
    n_events, n_components = size(weights)
    
    if length(yields) != n_components
        throw(ArgumentError("Number of yields must match number of components"))
    end
    
    # Calculate sums for each component
    sums = [sum(weights[:, i]) for i in 1:n_components]
    
    # Calculate relative errors
    relative_errors = abs.((sums .- yields) ./ yields)
    
    # Check if all components pass the tolerance test
    passed = all(relative_errors .< rtol)
    
    return (
        passed = passed,
        relative_errors = relative_errors,
        sums = sums
    )
end
