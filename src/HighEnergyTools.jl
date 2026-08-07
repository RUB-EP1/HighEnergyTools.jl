module HighEnergyTools

using Distributions
using LinearAlgebra
using QuadGK

export sPlot, sWeights
export wMatrix, inv_W
export sWeights_vector_with_variance
export check_wMatrix_condition

include("sweights.jl")

end # module HighEnergyTools
