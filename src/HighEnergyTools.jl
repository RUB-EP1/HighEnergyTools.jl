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
using Distributions
using LinearAlgebra
using DataFrames
using ForwardDiff
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
export chi2
export nll, fit_nll
include("fitting.jl")

export Anka, Frida
export build_model
include("models.jl")

export find_zero_two_sides
export interpolate_to_zero
export support_union
include("utils.jl")

export WithData, curvehistpulls
include("plotting_recipe.jl")

export sPlot, sWeights
export wMatrix, inv_W
export sWeights_vector, sWeights_covariance
export sWeights_vector_with_variance
export check_wMatrix_condition
export fit_and_sWeights
export sWeights_dataframe
export plot_sWeighted_histogram
include("sweights.jl")

end # module HighEnergyTools
