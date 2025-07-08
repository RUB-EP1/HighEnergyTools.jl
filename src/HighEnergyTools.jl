module HighEnergyTools

using QuadGK, Parameters
using Random, Statistics
using FHist
using Optim
using Distributions
using LinearAlgebra
using DataFrames
using ForwardDiff
#
using Plots, RecipesBase
using Plots.PlotMeasures: mm

export gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled
include("functions.jl")

export sample_rejection, sample_inversion
include("sampling.jl")

export fit_enll, extended_nll
export chi2
include("fitting.jl")

export Anka, Frida
export peak1_func, peak2_func
export background_func
export total_func
include("models.jl")

export find_zero_two_sides
export interpolate_to_zero
export support_union
include("utils.jl")

export WithData, curvehistpulls
include("plotting_recipe.jl")

export Wmatrix, sWeights
export sWeights_vector, sWeights_covariance
export sWeights_vector_with_variance
export check_Wmatrix_condition
export fit_and_sWeights
export sWeights_dataframe
export plot_sWeighted_histogram
include("sweights.jl")

end # module HighEnergyTools
