
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
