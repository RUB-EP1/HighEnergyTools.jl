# src/test_sweights.jl

using Test
using HighEnergyTools
using Distributions
using LinearAlgebra
using QuadGK

@testset "Wmatrix with explicit functions" begin
    dS(x) = pdf(Normal(0, 1), x)
    dB(x) = pdf(Normal(5, 1), x)
    f(x) = 0.5 * dS(x) + 0.5 * dB(x)
    lims = (-5, 10)
    W = Wmatrix(dS, dB, f, lims)
    @test size(W) == (2, 2)
    @test isapprox(W, W', atol = 1e-8)  # symmetric
    @test all(diag(W) .> 0)
end

@testset "Wmatrix with MixtureModels" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1)], [1.0])
    f(x) = 0.3 * pdf(pdfS, x) + 0.7 * pdf(pdfB, x)
    W1 = Wmatrix(pdfS, pdfB, f)
    dS(x) = pdf(pdfS, x)
    dB(x) = pdf(pdfB, x)
    lims = (-10.0, 15.0)
    W2 = Wmatrix(dS, dB, f, lims)
    @test size(W1) == (2, 2)
    @test isapprox(W1, W1', atol = 1e-8)
    @test isapprox(W1, W2, rtol = 1e-6)
end

@testset "sWeights basic properties" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1)], [1.0])
    f_sig = 0.4
    wS, wB = sWeights(pdfS, pdfB, f_sig)
    # Near pure signal region
    @test wS(0.0) > 0.9
    @test wB(0.0) < 0.1
    # Near pure background region
    @test wB(5.0) > 0.9
    @test wS(5.0) < 0.1
    # sWeights sum to 1 (approximately)
    tol = 0.02
    for x in [-2.0, 0.0, 2.0, 5.0, 7.0]
        @test isapprox(wS(x) + wB(x), 1.0; atol = 1e-8)
        @test -tol <= wS(x) <= 1.0 + tol
        @test -tol <= wB(x) <= 1.0 + tol
    end
end


@testset "sWeights edge cases" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1)], [1.0])
    tol = 0.02
    # All signal
    wS, wB = sWeights(pdfS, pdfB, 1.0)
    @test all(isapprox(wS(x), 1.0; atol = tol) for x = -3.0:1.0:3.0)
    @test all(isapprox(wB(x), 0.0; atol = tol) for x = -3.0:1.0:3.0)
    # All background
    wS, wB = sWeights(pdfS, pdfB, 0.0)
    @test all(isapprox(wS(x), 0.0; atol = tol) for x = 3.0:1.0:7.0)
    @test all(isapprox(wB(x), 1.0; atol = tol) for x = 3.0:1.0:7.0)
    # Overlapping distributions
    pdfS2 = MixtureModel([Normal(0, 1)], [1.0])
    pdfB2 = MixtureModel([Normal(0.5, 1)], [1.0])
    wS, wB = sWeights(pdfS2, pdfB2, 0.5)
    @test all(isapprox(wS(x) + wB(x), 1.0; atol = 1e-8) for x = -2.0:0.5:2.0)
end

@testset "sWeights features" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
    nS, nB = 40, 60
    xs = [-2.0, 0.0, 5.0, 8.0]

    # Test sWeights for yildsyields
    wS, wB = sWeights(pdfS, pdfB, nS, nB)
    @test isapprox(wS(0.0), 1.0; atol = 0.1)
    @test isapprox(wB(5.0), 1.0; atol = 0.1)

    # Test covariance matrix
    cov = sWeights_covariance(pdfS, pdfB, nS / (nS + nB))
    @test size(cov) == (2, 2)
    @test cov[1, 1] > 0

    # Test vectorized weights
    ws, wb = sWeights_vector(pdfS, pdfB, nS / (nS + nB), xs)
    @test length(ws) == length(xs)
    @test ws[2] > ws[1] - 0.01 # signal weight higher near signal mean

    # Test condition number
    f(x) = (nS / (nS + nB)) * pdf(pdfS, x) + (nB / (nS + nB)) * pdf(pdfB, x)
    W = Wmatrix(pdfS, pdfB, f)
    condW = check_Wmatrix_condition(W)
    @test condW > 0
end

@testset "sWeights_vector_with_variance" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
    xs = [-2.0, 0.0, 5.0, 8.0]
    ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, 0.4, xs)
    @test length(ws) == length(xs)
    @test length(vs) == length(xs)
    @test all(vs .>= 0)
    @test all(vb .>= 0)
    # sWeights should sum to about 1 for pure regions
    @test ws[2] ≈ 1.0 atol = 0.1 # near pure signal
    @test wb[3] ≈ 1.0 atol = 0.1 # near pure background
end

@testset "fit_and_sWeights" begin
    pdfS = MixtureModel([Normal(0, 1)], [1.0])
    pdfB = MixtureModel([Normal(5, 1.5)], [1.0])
    data = vcat(rand(pdfS, 40), rand(pdfB, 60))
    result, nS, nB, cov, ws, wb, vs, vb = fit_and_sWeights(pdfS, pdfB, data)
    @test abs(nS + nB - length(data)) < 1.0
    @test all(vs .>= 0)
    @test all(vb .>= 0)
    @test length(ws) == length(data)
end
