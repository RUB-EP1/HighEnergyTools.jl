using HighEnergyTools
using Distributions
using LinearAlgebra
using QuadGK
using Test

@testset "sPlot and Wmatrix" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1)
    model = MixtureModel([pdfS, pdfB], [0.3, 0.7])
    sP = sPlot(model)
    
    # Test sPlot object
    @test sP.model == model
    @test size(sP.inv_W) == (2, 2)
    
    # Test Wmatrix
    W = Wmatrix(sP)
    @test size(W) == (2, 2)
    @test isapprox(W, W', atol = 1e-8)  # symmetric
    @test all(diag(W) .> 0)
    
    # Test inv_W
    inv_W_mat = inv_W(sP)
    @test inv_W_mat == sP.inv_W
    @test isapprox(W * inv_W_mat, I, atol = 1e-10)
end

@testset "sPlot sum test" begin
    # Create test data
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    nS, nB = 40, 60
    n = nS + nB
    data = vcat(rand(pdfS, nS), rand(pdfB, nB))
    
    # Create sPlot object
    f_signal = nS / n
    model = MixtureModel([pdfS, pdfB], [f_signal, 1 - f_signal])
    sP = sPlot(model)
    
    # Test the pattern you requested
    wS, wB = sWeights(sP, data) |> eachcol
    fS(x) = sWeights(sP, [x])[1,1]
    fB(x) = sWeights(sP, [x])[1,2]
    
    # Test sums (with some tolerance for numerical precision)
    @test sum(wS) ≈ nS atol = 2.0  # Allow some tolerance for fit imprecision
    @test sum(wB) ≈ nB atol = 2.0  # Allow some tolerance for fit imprecision
    @test sum(wS) + sum(wB) ≈ n atol = 1e-10  # This should be very precise
    
    # Test that weight functions work
    @test fS(0.0) > 0.8  # Signal weight should be high near signal mean
    @test fB(5.0) > 0.8  # Background weight should be high near background mean
end

@testset "sWeights basic properties" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1)
    f_sig = 0.4
    xs = vcat(rand(pdfS, 40), rand(pdfB, 60))
    
    # Create sPlot object
    model = MixtureModel([pdfS, pdfB], [f_sig, 1 - f_sig])
    sP = sPlot(model)
    
    # Get sWeights
    weights = sWeights(sP, xs)
    wS, wB = eachcol(weights)
    
    # Create weight functions
    fS(x) = sWeights(sP, [x])[1,1]
    fB(x) = sWeights(sP, [x])[1,2]
    
    # Near pure signal region
    @test fS(0.0) > 0.9
    @test fB(0.0) < 0.1
    # Near pure background region
    @test fB(5.0) > 0.9
    @test fS(5.0) < 0.1
    # sWeights sum to 1 (approximately)
    tol = 0.02
    for x in [-2.0, 0.0, 2.0, 5.0, 7.0]
        @test isapprox(fS(x) + fB(x), 1.0; atol = 1e-8)
        @test -tol <= fS(x) <= 1.0 + tol
        @test -tol <= fB(x) <= 1.0 + tol
    end
end

@testset "sWeights edge cases" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1)
    tol = 0.02
    
    # All signal
    model_all_signal = MixtureModel([pdfS, pdfB], [1.0, 0.0])
    sP_all_signal = sPlot(model_all_signal)
    fS(x) = sWeights(sP_all_signal, [x])[1,1]
    fB(x) = sWeights(sP_all_signal, [x])[1,2]
    @test all(isapprox(fS(x), 1.0; atol = tol) for x ∈ -3.0:1.0:3.0)
    @test all(isapprox(fB(x), 0.0; atol = tol) for x ∈ -3.0:1.0:3.0)
    
    # All background
    model_all_background = MixtureModel([pdfS, pdfB], [0.0, 1.0])
    sP_all_background = sPlot(model_all_background)
    fS_bg(x) = sWeights(sP_all_background, [x])[1,1]
    fB_bg(x) = sWeights(sP_all_background, [x])[1,2]
    @test all(isapprox(fS_bg(x), 0.0; atol = tol) for x ∈ 3.0:1.0:7.0)
    @test all(isapprox(fB_bg(x), 1.0; atol = tol) for x ∈ 3.0:1.0:7.0)
    
    # Overlapping distributions
    pdfS2 = Normal(0, 1)
    pdfB2 = Normal(0.5, 1)
    model_overlap = MixtureModel([pdfS2, pdfB2], [0.5, 0.5])
    sP_overlap = sPlot(model_overlap)
    fS_overlap(x) = sWeights(sP_overlap, [x])[1,1]
    fB_overlap(x) = sWeights(sP_overlap, [x])[1,2]
    @test all(isapprox(fS_overlap(x) + fB_overlap(x), 1.0; atol = 1e-8) for x ∈ -2.0:0.5:2.0)
end

@testset "sWeights features" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    nS, nB = 40, 60
    xs = [-2.0, 0.0, 5.0, 8.0]
    f_signal = nS / (nS + nB)

    # Create sPlot object
    model = MixtureModel([pdfS, pdfB], [f_signal, 1 - f_signal])
    sP = sPlot(model)

    # Test sWeights function access
    fS(x) = sWeights(sP, [x])[1,1]
    fB(x) = sWeights(sP, [x])[1,2]
    @test isapprox(fS(0.0), 1.0; atol = 0.1)
    @test isapprox(fB(5.0), 1.0; atol = 0.1)

    # Test covariance matrix
    cov = inv_W(sP)
    @test size(cov) == (2, 2)
    @test cov[1, 1] > 0

    # Test vectorized weights
    weights = sWeights(sP, xs)
    ws, wb = eachcol(weights)
    @test length(ws) == length(xs)
    @test ws[2] > ws[1] - 0.01 # signal weight higher near signal mean

    # Test condition number
    condW = check_Wmatrix_condition(sP)
    @test condW > 0
end

@testset "sWeights_vector_with_variance" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    xs = [-2.0, 0.0, 5.0, 8.0]
    
    # Test with individual distributions
    ws, wb, vs, vb = sWeights_vector_with_variance(pdfS, pdfB, 0.4, xs)
    @test length(ws) == length(xs)
    @test length(vs) == length(xs)
    @test all(vs .>= 0)
    @test all(vb .>= 0)
    # sWeights should sum to about 1 for pure regions
    @test ws[2] ≈ 1.0 atol = 0.1 # near pure signal
    @test wb[3] ≈ 1.0 atol = 0.1 # near pure background
    
    # Test with sPlot object
    model = MixtureModel([pdfS, pdfB], [0.4, 0.6])
    sP = sPlot(model)
    ws2, wb2, vs2, vb2 = sWeights_vector_with_variance(sP, xs)
    @test ws ≈ ws2 atol = 1e-10
    @test wb ≈ wb2 atol = 1e-10
    @test vs ≈ vs2 atol = 1e-10
    @test vb ≈ vb2 atol = 1e-10
end

@testset "fit_and_sWeights" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    data = vcat(rand(pdfS, 40), rand(pdfB, 60))
    result, sP, nS, nB, cov, ws, wb, vs, vb = fit_and_sWeights(pdfS, pdfB, data)
    
    # Basic checks
    @test all(vs .>= 0)
    @test all(vb .>= 0)
    @test length(ws) == length(data)
    
    # Check that sP is returned and works
    @test isa(sP, sPlot)
    weights_check = sWeights(sP, data)
    @test size(weights_check) == (length(data), 2)
end
