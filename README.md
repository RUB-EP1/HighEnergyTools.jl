# HighEnergyTools.jl

Julia implementation of the **sPlot** method and **sWeights** for mixture models in high-energy physics.

Given a fitted `MixtureModel` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), this package computes per-event signal and background weights that preserve yield normalisation — the standard technique for projecting fitted components onto other observables.

Dependencies are minimal: **Distributions.jl** (for `MixtureModel`) and **QuadGK.jl** (for the weight-matrix integrals).

For PDF construction, sampling, and fitting, use the related ecosystem:

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) — standard probability distributions
- [DistributionsHEP.jl](https://github.com/JuliaHEP/DistributionsHEP.jl) — HEP-specific distributions and extended mixtures
- [NumericalDistributions.jl](https://github.com/mmikhasenko/NumericalDistributions.jl) — user-defined PDFs with automatic normalisation
- [FitUtils.jl](https://github.com/mmikhasenko/FitUtils.jl) — extended maximum-likelihood fitting

## Installation

The package is not registered. Install from GitHub:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/RUB-EP1/HighEnergyTools.jl.git"))
using HighEnergyTools
```

## Usage

```julia
using Distributions
using HighEnergyTools

pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
model = MixtureModel([pdfS, pdfB], [0.4, 0.6])

sP = sPlot(model)
data = vcat(rand(pdfS, 40), rand(pdfB, 60))
wS, wB = sWeights(sP, data) |> eachcol

# Per-event weight at a single point
fS(x) = sWeights(sP, [x])[1, 1]
```

Fit the mixture elsewhere (e.g. with FitUtils or Minuit2), then pass the fitted `MixtureModel` to `sPlot`.

## Testing

```julia
import Pkg
Pkg.test("HighEnergyTools")
```
