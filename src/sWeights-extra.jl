
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

