module HighEnergyTools

using NumericalDistributions
using DistributionsHEP
using Distributions
using Parameters
using Statistics
using QuadGK
using Random
using FHist
using Optim
#
using Plots, RecipesBase
using Plots.PlotMeasures: mm

import Distributions: pdf
export pdf

export gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled
include("functions.jl")

export sample_rejection, sample_inversion
include("sampling.jl")

export fit_enll, extended_nll
export nll, fit_nll
include("fitting.jl")

export Anka, Frida
export build_model
include("models.jl")

export find_zero_two_sides
export interpolate_to_zero
include("utils.jl")

export WithData, curvehistpulls
include("plotting_recipe.jl")

end # module HighEnergyTools
