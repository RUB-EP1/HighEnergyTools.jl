# sWeights example
#
# Compute sWeights from a fitted signal+background mixture.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions
using HighEnergyTools

pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
data = vcat(rand(pdfS, 400), rand(pdfB, 600))

# Yields from an extended fit (here: known counts for illustration)
nS, nB = 400.0, 600.0
model = MixtureModel([pdfS, pdfB], [nS, nB] ./ (nS + nB))

sP = sPlot(model)
ws, wb = sWeights(sP, data) |> eachcol

println("Sum of signal sWeights: ", sum(ws))
println("Sum of background sWeights: ", sum(wb))

fS(x) = sWeights(sP, [x])[1, 1]
println("Signal weight at x=0: ", fS(0.0))
