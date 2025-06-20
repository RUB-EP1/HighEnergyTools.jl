# src/test_sweights.jl

using Test
using Distributions
using QuadGK
include("sweights.jl")

@testset "Wmatrix with explicit functions" begin
    dS(x) = pdf(Normal(0,1), x)
    dB(x) = pdf(Normal(5,1), x)
    f(x) = 0.5*dS(x) + 0.5*dB(x)
    lims = (-5, 10)
    W = Wmatrix(dS, dB, f, lims)
    @test size(W) == (2,2)
    @test isapprox(W, W', atol=1e-8)  # symmetric
    @test all(diag(W) .> 0)
end

@testset "Wmatrix with MixtureModels" begin
    pdfS = MixtureModel([Normal(0,1)], [1.0])
    pdfB = MixtureModel([Normal(5,1)], [1.0])
    f(x) = 0.3*pdf(pdfS, x) + 0.7*pdf(pdfB, x)
    W1 = Wmatrix(pdfS, pdfB, f)
    dS(x) = pdf(pdfS, x)
    dB(x) = pdf(pdfB, x)
    lims = support_union(pdfS, pdfB)
    W2 = Wmatrix(dS, dB, f, lims)
    @test size(W1) == (2,2)
    @test isapprox(W1, W1', atol=1e-8)
    @test isapprox(W1, W2, rtol=1e-6)
end

@testset "sWeights basic properties" begin
    pdfS = MixtureModel([Normal(0,1)], [1.0])
    pdfB = MixtureModel([Normal(5,1)], [1.0])
    f_sig = 0.4
    wS, wB = sWeights(pdfS, pdfB, f_sig)
    # Near pure signal region
    @test wS(0.0) > 0.9
    @test wB(0.0) < 0.1
    # Near pure background region
    @test wB(5.0) > 0.9
    @test wS(5.0) < 0.1
    # sWeights sum to 1 (approximately)
    for x in [-2.0, 0.0, 2.0, 5.0, 7.0]
        @test isapprox(wS(x) + wB(x), 1.0; atol=1e-8)
        @test 0.0 <= wS(x) <= 1.0
        @test 0.0 <= wB(x) <= 1.0
    end
end

@testset "sWeights edge cases" begin
    pdfS = MixtureModel([Normal(0,1)], [1.0])
    pdfB = MixtureModel([Normal(5,1)], [1.0])
    # All signal
    wS, wB = sWeights(pdfS, pdfB, 1.0)
    @test all(wS(x) ≈ 1.0 for x in -3.0:1.0:3.0)
    @test all(wB(x) ≈ 0.0 for x in -3.0:1.0:3.0)
    # All background
    wS, wB = sWeights(pdfS, pdfB, 0.0)
    @test all(wS(x) ≈ 0.0 for x in 3.0:1.0:7.0)
    @test all(wB(x) ≈ 1.0 for x in 3.0:1.0:7.0)
    # Overlapping distributions
    pdfS2 = MixtureModel([Normal(0,1)], [1.0])
    pdfB2 = MixtureModel([Normal(0.5,1)], [1.0])
    wS, wB = sWeights(pdfS2, pdfB2, 0.5)
    @test all(0.0 <= wS(x) <= 1.0 for x in -2.0:0.5:2.0)
    @test all(0.0 <= wB(x) <= 1.0 for x in -2.0:0.5:2.0)
end