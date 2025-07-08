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
    ϵ = 1e-12  # Small threshold to avoid division by zero
    W = [
        quadgk(x -> di(x) * dj(x) / max(f(x), ϵ), lims...)[1]
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

"""
    sWeights(pdfS::MixtureModel, pdfB::MixtureModel, n_signal::Real, n_background::Real)

Compute the sWeight functions for signal and background components using the absolute fitted yields.

# Arguments
- `pdfS`: Signal model as a `MixtureModel`.
- `pdfB`: Background model as a `MixtureModel`.
- `n_signal`: Fitted number of signal events (must be non-negative).
- `n_background`: Fitted number of background events (must be non-negative).

# Returns
- A pair of functions `(w_signal, w_background)` such that:
    - `w_signal(x)`: sWeight for signal at point `x`
    - `w_background(x)`: sWeight for background at point `x`

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
nS, nB = 40, 60
wS, wB = sWeights(pdfS, pdfB, nS, nB)
println(wS(0.0))
```
"""
function sWeights(pdfS::MixtureModel, pdfB::MixtureModel, n_signal::Real, n_background::Real)
    N = n_signal + n_background
    f_signal = n_signal / N
    return sWeights(pdfS, pdfB, f_signal)
end

"""
    sWeights_covariance(pdfS::MixtureModel, pdfB::MixtureModel, fraction_signal::Real)

Return the covariance matrix of the fitted yields as used in the sPlot formalism.

# Arguments
- `pdfS`: Signal model as a `MixtureModel`.
- `pdfB`: Background model as a `MixtureModel`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).

# Returns
- `cov::Matrix{Float64}`: 2×2 covariance matrix of the yields (signal, background).

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
cov = sWeights_covariance(pdfS, pdfB, 0.4)
println(cov)
```
"""
function sWeights_covariance(pdfS::MixtureModel, pdfB::MixtureModel, fraction_signal::Real)
    f(x) = fraction_signal * pdf(pdfS, x) + (1 - fraction_signal) * pdf(pdfB, x)
    W = Wmatrix(pdfS, pdfB, f)
    return inv(W)
end

"""
    sWeights_vector(pdfS, pdfB, fraction_signal, xs)

Compute the sWeights for arrays of data points.

# Arguments
- `pdfS`: Signal model as a `MixtureModel`.
- `pdfB`: Background model as a `MixtureModel`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).
- `xs`: Vector of data points.

# Returns
- `(ws_signal, ws_background)`: Tuple of vectors containing the sWeights for signal and background at each point in `xs`.

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
xs = [-2.0, 0.0, 5.0, 8.0]
ws, wb = sWeights_vector(pdfS, pdfB, 0.4, xs)
println(ws)
```
"""
function sWeights_vector(pdfS, pdfB, fraction_signal, xs::AbstractVector)
    wS, wB = sWeights(pdfS, pdfB, fraction_signal)
    return (wS.(xs), wB.(xs))
end

"""
    check_Wmatrix_condition(W::AbstractMatrix)

Warn if the sPlot weight matrix is ill-conditioned, which may indicate numerical instability.

# Arguments
- `W`: The sPlot weight matrix (typically 2×2).

# Returns
- `condW::Float64`: The condition number of `W`.

# Example
```julia
W = [1.0 0.1; 0.1 1.0]
condW = check_Wmatrix_condition(W)
```
"""
function check_Wmatrix_condition(W::AbstractMatrix)
    condW = cond(W)
    if condW > 1e8
        @warn "Weight matrix is ill-conditioned (condition number = $condW). Results may be unstable."
    end
    return condW
end

"""
    sWeights_vector_with_variance(pdfS, pdfB, fraction_signal, xs)

Compute the sWeights and their variances for arrays of data points.

# Arguments
- `pdfS`: Signal model as a `MixtureModel`.
- `pdfB`: Background model as a `MixtureModel`.
- `fraction_signal`: Estimated fraction of signal in the total data (between 0 and 1).
- `xs`: Vector of data points.

# Returns
- `(ws_signal, ws_background, var_signal, var_background)`: Tuple of vectors containing the sWeights and their variances for signal and background at each point in `xs`.

# Notes
The variance is computed according to the sPlot formalism:
``\\mathrm{Var}[w_i(x)] = \\sum_{j,k} V_{ij} V_{ik} \\frac{p_j(x) p_k(x)}{f(x)^2}``
where ``V`` is the covariance matrix of the yields, ``p_j(x)`` are the PDFs, and ``f(x)`` is the total PDF.

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
xs = [-2.0, 0.0, 5.0, 8.0]
ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, 0.4, xs)
println(ws)
println(vs)
```
"""
function sWeights_vector_with_variance(pdfS, pdfB, fraction_signal, xs::AbstractVector)
    # PDFs
    pS(x) = pdf(pdfS, x)
    pB(x) = pdf(pdfB, x)
    f(x) = fraction_signal * pS(x) + (1 - fraction_signal) * pB(x)
    # Covariance matrix
    V = sWeights_covariance(pdfS, pdfB, fraction_signal)
    # sWeight coefficients (alpha)
    W = Wmatrix(pdfS, pdfB, f)
    α = inv(W)
    αS = α[:, 1] ./ abs(sum(α[:, 1]))
    αB = α[:, 2] ./ abs(sum(α[:, 2]))
    # sWeight functions
    numerator_s(x) = αS[1] * pS(x) + αS[2] * pB(x)
    numerator_b(x) = αB[1] * pS(x) + αB[2] * pB(x)
    weight_signal(x) = numerator_s(x) / f(x) * fraction_signal
    weight_background(x) = numerator_b(x) / f(x) * (1 - fraction_signal)
    # Variance functions
    function variance(i, x)
        # i = 1 for signal, 2 for background
        ps = [pS(x), pB(x)]
        v = 0.0
        for j in 1:2, k in 1:2
            v += V[i, j] * V[i, k] * ps[j] * ps[k]
        end
        return v / (f(x)^2)
    end
    var_signal(x) = variance(1, x)
    var_background(x) = variance(2, x)
    # Vectorized
    ws = weight_signal.(xs)
    wb = weight_background.(xs)
    vs = var_signal.(xs)
    vb = var_background.(xs)
    return (ws, wb, vs, vb)
end

"""
    fit_and_sWeights(pdfS, pdfB, data; support=nothing, init_fsig=0.5)

Fit the mixture model (signal + background) to the data using extended NLL,
then compute sWeights and their variances for each event.

# Arguments
- `pdfS`: Signal MixtureModel (shape fixed).
- `pdfB`: Background MixtureModel (shape fixed).
- `data`: Vector of data points.
- `support`: Optional tuple for fit range. Defaults to extrema(data).
- `init_fsig`: Initial guess for signal fraction.

# Returns
- `result`: Optim.jl fit result.
- `n_signal`, `n_background`: Fitted yields.
- `cov`: Covariance matrix of yields.
- `ws`, `wb`: sWeights for signal and background (vectors).
- `vs`, `vb`: Variances of sWeights (vectors).

# Example
```julia
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
data = vcat(rand(pdfS, 40), rand(pdfB, 60))
result, nS, nB, cov, ws, wb, vs, vb = fit_and_sWeights(pdfS, pdfB, data)
```
"""
function fit_and_sWeights(pdfS, pdfB, data; support=nothing, init_fsig=0.5)
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
    result = fit_enll(model, init_pars, data; support=support)

    nS, nB = Optim.minimizer(result) |> Tuple{Float64,Float64}

    # Redefine the NLL objective for Hessian calculation
    nll(p) = extended_nll(model, p, data; support=support)

    # Covariance estimate (inverse Hessian)
    hess = ForwardDiff.hessian(nll, [nS, nB])
    cov = inv(hess)

    # Compute sWeights and variances
    f_signal = nS / (nS + nB)
    ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, f_signal, data)

    return result, nS, nB, cov, ws, wb, vs, vb
end

"""
    sWeights_dataframe(pdfS, pdfB, fraction_signal, xs)

Return a DataFrame with data, sWeights, and variances for each event.

# Arguments
- `pdfS`, `pdfB`: Signal and background MixtureModels.
- `fraction_signal`: Estimated signal fraction.
- `xs`: Vector of data points.

# Returns
- `df`: DataFrame with columns: :x, :ws_signal, :ws_background, :var_signal, :var_background

# Example
```julia
using DataFrames
pdfS = MixtureModel([Normal(0, 1)], [1.0])
pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
xs = [-2.0, 0.0, 5.0, 8.0]
df = sWeights_dataframe(pdfS, pdfB, 0.4, xs)
```
"""
function sWeights_dataframe(pdfS, pdfB, fraction_signal, xs)
    ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, fraction_signal, xs)
    return DataFrame(x=xs, ws_signal=ws, ws_background=wb, var_signal=vs, var_background=vb)
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
function plot_sWeighted_histogram(xs, weights; variances=nothing, nbins=30, label="sWeighted", xlabel="x", ylabel="Events", color=:blue)
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
        bar(bin_centers, counts, yerr=errs, label=label, xlabel=xlabel, ylabel=ylabel, color=color)
    else
        bar(bin_centers, counts, label=label, xlabel=xlabel, ylabel=ylabel, color=color)
    end
end