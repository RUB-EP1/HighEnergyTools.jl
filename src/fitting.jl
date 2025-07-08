_quadgk_call(f, support) = quadgk(f, support...)[1]

function _data_in_support(data, support)
    red_data = filter(x -> support[1] < x < support[2], data)
    if length(data) > length(red_data)
        @info "Reduced dataset to domain $support to get normalizations right"
    end
    return red_data
end


"""
    extended_nll(model, parameters, data; support = extrema(data), normalization_call = _quadgk_call)

Calculate the extended negative log likelihood (ENLL) for a given model and dataset, passing parameters of the model explicitly.

# Arguments
- `model`: A function that represents the model. It should take two arguments: observable and parameters.
- `parameters`: The parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated. Defaults to the range of the data.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `extended_nll_value`: The extended negative log likelihood value for the given model and data.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
parameters = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

enll_value = extended_nll(model, parameters, data; support=support)
```
"""
extended_nll(
    model,
    parameters,
    data;
    support = extrema(data),
    normalization_call = _quadgk_call,
) = extended_nll(x -> model(x, parameters), data; support, normalization_call)

"""
    extended_nll(model, data; support = extrema(data), normalization_call = _quadgk_call)

Calculate the extended negative log likelihood (ENLL) for a given model and dataset.

# Arguments
- `model`: A function that represents the model. It should take two arguments: observable and parameters.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated. Defaults to the range of the data.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `extended_nll_value`: The extended negative log likelihood value for the given model and data.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

enll_value = extended_nll(model, data)
```
"""
function extended_nll(
    model,
    data;
    support = extrema(data),
    normalization_call = _quadgk_call,
)
    # negative log likelihood for model
    minus_sum_log = -sum(data) do x
        value = model(x)
        value > 0 ? log(value) : -1e10
    end
    # extended negative log likelihood
    return minus_sum_log + normalization_call(model, support)
end

"""
    fit_enll(model, init_pars, data; support = extrema(data), alg=BFGS, normalization_call = _quadgk_call)

Fit the model parameters using the extended negative log likelihood (ENLL) method.

# Arguments
- `model`: A function that represents the model to be fitted. It should take two arguments: observable and parameters.
- `init_pars`: Initial parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated.
- `alg`: (Optional) Optimization algorithm to be used. Default is `BFGS`.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `result`: The optimization result that minimizes the extended negative log likelihood.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
init_pars = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

fit_result = fit_enll(model, init_pars, data, support)
```
"""
function fit_enll(
    model,
    init_pars,
    data;
    support = extrema(data),
    alg = BFGS(),
    normalization_call = _quadgk_call,
)
    # check if data is within limits
    filtered_data = _data_in_support(data, support)
    # fit the model parameters
    optimize(
        p -> extended_nll(
            model,
            typeof(init_pars)(p),
            filtered_data;
            support = support,
            normalization_call = normalization_call,
        ),
        collect(init_pars),
        alg,
    )
end

"""
    chi2(h, model)

Compute the binned χ² statistic between a histogram `h` and a model function.

- `h`: Histogram object (must support `bincenters` and `bincounts`).
- `model`: Function mapping bin centers to expected counts.

Returns the sum over bins of (expected - observed)^2 / observed, skipping bins with zero observed counts.

# Example
```julia
using StatsBase
h = fit(Histogram, randn(100), 20)
model(x) = 5 * exp(-x^2/2)
chi2(h, model)
```
"""
function chi2(h, model)
    xv = bincenters(h)
    yv = bincounts(h)
    yv_pred = model.(xv)
    # Avoid division by zero: skip bins with zero observed counts
    mask = yv .> 0
    return sum(@. (yv_pred[mask] - yv[mask])^2 / yv[mask])
end

"""
    chi2(h, d::UnivariateDistribution)

Compute the binned χ² statistic between a histogram `h` and a univariate distribution `d`.

- `h`: Histogram object (must support `bincenters`, `bincounts`, and `integral`).
- `d`: A `UnivariateDistribution` from Distributions.jl.

Scales the PDF of `d` to match the total counts in `h` before computing χ².

# Example
```julia
using StatsBase, Distributions
h = fit(Histogram, rand(Normal(), 100), 20)
d = Normal()
chi2(h, d)
```
"""
function chi2(h, d::UnivariateDistribution)
    scale = FHist.integral(h; width = true)
    f(x) = pdf(d, x) * scale
    chi2(h, f)
end
