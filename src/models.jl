"""
    SpectrumModel

Abstract type representing a spectrum model for high-energy physics analysis.
All concrete spectrum models should subtype this abstract type.
"""
abstract type SpectrumModel end


"""
    build_two_component_model(pars; build_signal, build_backgr)

Build a two-component mixture model from signal and background components.

# Arguments
- `pars`: Named tuple containing model parameters
- `build_signal`: Function that constructs the signal component from signal parameters
- `build_backgr`: Function that constructs the background component from background parameters

# Returns
- `MixtureModel`: A mixture model with signal and background components

# Parameters
- `sig`: Signal parameters
- `bgd`: Background parameters
- `logfB`: Log of the background fraction
"""
function build_two_component_model(pars; build_signal, build_backgr)
    @unpack sig, bgd, logfB = pars
    prior_unnorm = [one(logfB), exp(logfB)]
    prior = prior_unnorm ./ sum(prior_unnorm)
    signal_model = build_signal(sig)
    background_model = build_backgr(bgd)
    return MixtureModel([signal_model, background_model], prior)
end


"""
    Anka{P <: Real} <: SpectrumModel
    Anka((a, b))
    Anka(a, b)

A spectrum model builder with a single signal component and a Chebyshev polynomial background.

# Fields
- `support::Tuple{P, P}`: The support interval (a, b) for the spectrum

Creates an Anka builder with support from `a` to `b`.

# Example
```julia
anka = Anka(1.1, 3.3)
pars = (sig = (μ = 2.2, σ = 0.06), bgd = (coeffs = [1.5, 1.1],), logfB = 0.0)
model = build_model(anka, pars)
```
"""
@with_kw struct Anka{P <: Real} <: SpectrumModel
    support::Tuple{P, P}
end
Anka(a, b) = Anka((a, b))

"""
    scaled_chebyshev(coeffs, support)

Create a scaled Chebyshev polynomial distribution over the specified support interval.

# Arguments
- `coeffs`: Coefficients for the Chebyshev polynomial
- `support`: Tuple (a, b) defining the support interval

# Returns
- `LocationScale`: A scaled Chebyshev distribution
"""
function scaled_chebyshev(coeffs, support)
    shift = (support[2] + support[1]) / 2
    scale = (support[2] - support[1]) / 2
    cheb = Chebyshev(coeffs, -1, 1)
    return cheb * scale + shift
end

"""
    build_model(m::Anka, pars)

Build an Anka spectrum model with signal and background components.

# Arguments
- `m::Anka`: The Anka model instance
- `pars`: Named tuple containing model parameters

# Returns
- `MixtureModel`: A mixture model with truncated normal signal and Chebyshev background

# Parameters
- `sig`: Signal parameters with fields `μ` (mean) and `σ` (standard deviation)
- `bgd`: Background parameters with field `coeffs` (Chebyshev coefficients)
- `logfB`: Log of the background fraction
"""
function build_model(m::Anka, pars)
    build_signal(sig) = truncated(Normal(sig.μ, sig.σ), m.support...)
    build_backgr(bgd) = scaled_chebyshev(bgd.coeffs, m.support)
    build_two_component_model(pars; build_signal, build_backgr)
end



"""
    Frida{P <: Real} <: SpectrumModel
    Frida((a, b))
    Frida(a, b)

A spectrum model with two signal components and a Chebyshev polynomial background.

# Fields
- `support::Tuple{P, P}`: The support interval (a, b) for the spectrum

Creates a Frida model with support from `a` to `b`.

# Example
```julia
frida = Frida(1.1, 3.3)
pars = (
    sig1 = (μ = 2.1, σ = 0.05),
    sig2 = (μ = 2.4, σ = 0.08),
    bgd = (coeffs = [1.5, 1.1],),
    logfS1 = -1.0,
    logfS2 = -2.0,
)
model = build_model(frida, pars)
```
"""
@with_kw struct Frida{P <: Real} <: SpectrumModel
    support::Tuple{P, P}
end
Frida(a, b) = Frida((a, b))

"""
    build_model(m::Frida, pars)

Build a Frida spectrum model builder with two truncated normal signal components and a Chebyshev polynomial background.

# Arguments
- `m::Frida`: The Frida builder instance
- `pars`: Named tuple containing model parameters

# Returns
- `MixtureModel`: A mixture model with two truncated normal signals and Chebyshev background

# Parameters
- `sig1`: First signal parameters with fields `μ` (mean) and `σ` (standard deviation)
- `sig2`: Second signal parameters with fields `μ` (mean) and `σ` (standard deviation)
- `bgd`: Background parameters with field `coeffs` (Chebyshev coefficients)
- `logfS1`: Log of the first signal fraction
- `logfS2`: Log of the second signal fraction
"""
function build_model(m::Frida, pars)
    @unpack sig1, sig2, bgd = pars
    @unpack logfS1, logfS2 = pars
    #
    prior_unnorm = [exp(logfS1), exp(logfS2), one(logfS1)]
    prior = prior_unnorm ./ sum(prior_unnorm)
    #
    s1_model = truncated(Normal(sig1.μ, sig1.σ), m.support...)
    s2_model = truncated(Normal(sig2.μ, sig2.σ), m.support...)
    background_model = scaled_chebyshev(bgd.coeffs, m.support)
    return MixtureModel([s1_model, s2_model, background_model], prior)
end
