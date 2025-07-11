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
    fit_and_sWeights(pdfS, pdfB, data; support=nothing, init_fsig=0.5)

Fit the mixture model (signal + background) to the data using extended NLL,
then compute sWeights and their variances for each event.

# Arguments
- `pdfS`: Signal UnivariateDistribution (shape fixed).
- `pdfB`: Background UnivariateDistribution (shape fixed).
- `data`: Vector of data points.
- `support`: Optional tuple for fit range. Defaults to extrema(data).
- `init_fsig`: Initial guess for signal fraction.

# Returns
- `result`: Optim.jl fit result.
- `sP`: The fitted sPlot object.
- `n_signal`, `n_background`: Fitted yields.
- `cov`: Covariance matrix of yields.
- `ws`, `wb`: sWeights for signal and background (vectors).
- `vs`, `vb`: Variances of sWeights (vectors).

# Example
```julia
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
data = vcat(rand(pdfS, 40), rand(pdfB, 60))
result, sP, nS, nB, cov, ws, wb, vs, vb = fit_and_sWeights(pdfS, pdfB, data)
# Use sP for further sWeight calculations:
wS, wB = sWeights(sP, data) |> eachcol
fS(x) = sWeights(sP, [x])[1,1]
fB(x) = sWeights(sP, [x])[1,2]
```
"""
function fit_and_sWeights(pdfS, pdfB, data; support = nothing, init_fsig = 0.5)
    N = length(data)
    support = isnothing(support) ? extrema(data) : support

    # Model: nS * pdfS(x) + nB * pdfB(x)
    function model(x, p)
        nS, nB = p
        nS * pdf(pdfS, x) + nB * pdf(pdfB, x)
    end

    # Initial guess: split total events by init_fsig
    init_nS = N * init_fsig
    init_nB = N - init_nS
    init_pars = [init_nS, init_nB]

    # Fit using extended NLL
    result = fit_enll(model, init_pars, data; support = support)
    nS, nB = Optim.minimizer(result) |> Tuple{Float64, Float64}

    # Create sPlot object with fitted fractions
    f_signal = nS / (nS + nB)
    model_fitted = MixtureModel([pdfS, pdfB], [f_signal, 1 - f_signal])
    sP = sPlot(model_fitted)

    # Covariance estimate (inverse Hessian)
    nll(p) = extended_nll(model, p, data; support = support)
    hess = ForwardDiff.hessian(nll, [nS, nB])
    cov = inv(hess)

    # Compute sWeights and variances using the sPlot object
    ws, wb, vs, vb = sWeights_vector_with_variance(sP, data)

    return result, sP, nS, nB, cov, ws, wb, vs, vb
end

"""
    sWeights_dataframe(sP::sPlot, xs)

Return a DataFrame with data, sWeights, and variances for each event using the sPlot object.

# Arguments
- `sP`: An `sPlot` object.
- `xs`: Vector of data points.

# Returns
- `df`: DataFrame with columns: :x, :ws_signal, :ws_background, :var_signal, :var_background

# Example
```julia
using DataFrames
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
model = MixtureModel([pdfS, pdfB], [0.4, 0.6])
sP = sPlot(model)
xs = [-2.0, 0.0, 5.0, 8.0]
df = sWeights_dataframe(sP, xs)
```
"""
function sWeights_dataframe(sP::sPlot, xs)
    ws, wb, vs, vb = sWeights_vector_with_variance(sP, xs)
    return DataFrame(
        x = xs,
        ws_signal = ws,
        ws_background = wb,
        var_signal = vs,
        var_background = vb,
    )
end

"""
    sWeights_dataframe(pdfS, pdfB, fraction_signal, xs)

Return a DataFrame with data, sWeights, and variances for each event using individual distributions.

# Arguments
- `pdfS`, `pdfB`: Signal and background UnivariateDistributions.
- `fraction_signal`: Estimated signal fraction.
- `xs`: Vector of data points.

# Returns
- `df`: DataFrame with columns: :x, :ws_signal, :ws_background, :var_signal, :var_background

# Example
```julia
using DataFrames
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
xs = [-2.0, 0.0, 5.0, 8.0]
df = sWeights_dataframe(pdfS, pdfB, 0.4, xs)
```
"""
function sWeights_dataframe(pdfS, pdfB, fraction_signal, xs)
    model = MixtureModel([pdfS, pdfB], [fraction_signal, 1 - fraction_signal])
    sP = sPlot(model)
    return sWeights_dataframe(sP, xs)
end

"""
    plot_sWeighted_histogram(xs, weights; variances=nothing, nbins=30, label="sWeighted", xlabel="x", ylabel="Events", color=:blue)

Plot a weighted histogram with optional error bars from sWeights.

# Arguments
- `xs`: Data points.
- `weights`: sWeights for each point.
- `variances`: Optional variances for error bars.
- `nbins`: Number of histogram bins.
- `label`, `xlabel`, `ylabel`, `color`: Plot styling options.

# Example
```julia
using Plots
xs = randn(100)
ws = ones(100)
plot_sWeighted_histogram(xs, ws)
```
"""
function plot_sWeighted_histogram(
    xs,
    weights;
    variances = nothing,
    nbins = 30,
    label = "sWeighted",
    xlabel = "x",
    ylabel = "Events",
    color = :blue,
)
    h = fit(Histogram, xs, nbins)
    bin_edges = h.edges[1]
    bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) ./ 2
    bin_indices = searchsortedlast.(Ref(bin_edges), xs)
    counts = zeros(length(bin_centers))
    errs = zeros(length(bin_centers))
    for (i, x) in enumerate(xs)
        b = bin_indices[i] - 1
        if 1 <= b <= length(counts)
            counts[b] += weights[i]
            if variances !== nothing
                errs[b] += variances[i]
            end
        end
    end
    if variances !== nothing
        errs = sqrt.(errs)
        bar(
            bin_centers,
            counts,
            yerr = errs,
            label = label,
            xlabel = xlabel,
            ylabel = ylabel,
            color = color,
        )
    else
        bar(
            bin_centers,
            counts,
            label = label,
            xlabel = xlabel,
            ylabel = ylabel,
            color = color,
        )
    end
end
