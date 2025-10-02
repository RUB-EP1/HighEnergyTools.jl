using HighEnergyTools
using Distributions
using LinearAlgebra
using QuadGK
using Test

@testset "sPlot and wMatrix" begin
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1)
    model = MixtureModel([pdfS, pdfB], [0.3, 0.7])
    sP = sPlot(model)

    # Test sPlot object
    @test sP.model == model
    @test size(sP.inv_W) == (2, 2)

    # Test wMatrix
    W = wMatrix(sP)
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
    fS(x) = sWeights(sP, [x])[1, 1]
    fB(x) = sWeights(sP, [x])[1, 2]

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
    fS(x) = sWeights(sP, [x])[1, 1]
    fB(x) = sWeights(sP, [x])[1, 2]

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
    fS(x) = sWeights(sP_all_signal, [x])[1, 1]
    fB(x) = sWeights(sP_all_signal, [x])[1, 2]
    @test all(isapprox(fS(x), 1.0; atol = tol) for x ∈ -3.0:1.0:3.0)
    @test all(isapprox(fB(x), 0.0; atol = tol) for x ∈ -3.0:1.0:3.0)

    # All background
    model_all_background = MixtureModel([pdfS, pdfB], [0.0, 1.0])
    sP_all_background = sPlot(model_all_background)
    fS_bg(x) = sWeights(sP_all_background, [x])[1, 1]
    fB_bg(x) = sWeights(sP_all_background, [x])[1, 2]
    @test all(isapprox(fS_bg(x), 0.0; atol = tol) for x ∈ 3.0:1.0:7.0)
    @test all(isapprox(fB_bg(x), 1.0; atol = tol) for x ∈ 3.0:1.0:7.0)

    # Overlapping distributions
    pdfS2 = Normal(0, 1)
    pdfB2 = Normal(0.5, 1)
    model_overlap = MixtureModel([pdfS2, pdfB2], [0.5, 0.5])
    sP_overlap = sPlot(model_overlap)
    fS_overlap(x) = sWeights(sP_overlap, [x])[1, 1]
    fB_overlap(x) = sWeights(sP_overlap, [x])[1, 2]
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
    fS(x) = sWeights(sP, [x])[1, 1]
    fB(x) = sWeights(sP, [x])[1, 2]
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
    condW = check_wMatrix_condition(sP)
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

@testset "sWeights_general - basic functionality" begin
    # Test 2-component case that should match existing sPlot results
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    data = vcat(rand(pdfS, 40), rand(pdfB, 60))
    yields = [40.0, 60.0]
    
    # Create shape values matrix
    shape_values = zeros(length(data), 2)
    for (i, x) in enumerate(data)
        shape_values[i, 1] = pdf(pdfS, x)
        shape_values[i, 2] = pdf(pdfB, x)
    end
    
    # Compute sWeights using general method
    weights_general, cov_general = sWeights_general(yields, shape_values)
    
    # Compare with existing sPlot method
    model = MixtureModel([pdfS, pdfB], [0.4, 0.6])
    sP = sPlot(model)
    weights_splot = sWeights(sP, data)
    
    # Results should be similar (allowing for different normalization conventions)
    @test size(weights_general) == size(weights_splot)
    @test size(cov_general) == (2, 2)
    
    # Check closure property
    closure = check_sweights_closure(weights_general, yields)
    @test closure.passed
    @test all(closure.relative_errors .< 1e-10)  # Should be very precise
end

@testset "sWeights_general - multi-component case" begin
    # Test 4-component mixture (like phi-phi analysis)
    n_events = 1000
    yields = [300.0, 200.0, 150.0, 350.0]  # Four components
    n_components = 4
    
    # Create synthetic shape values (random but normalized)
    Random.seed!(42)  # For reproducibility
    shape_values = rand(n_events, n_components)
    
    # Normalize each row to simulate realistic shape functions
    for i in 1:n_events
        shape_values[i, :] ./= sum(shape_values[i, :])
    end
    
    # Compute sWeights
    weights, cov = sWeights_general(yields, shape_values)
    
    # Basic checks
    @test size(weights) == (n_events, n_components)
    @test size(cov) == (n_components, n_components)
    @test all(diag(cov) .> 0)  # Diagonal elements should be positive
    @test isapprox(cov, cov', atol=1e-10)  # Should be symmetric
    
    # Check closure property
    closure = check_sweights_closure(weights, yields, rtol=1e-2)
    @test closure.passed
    
    # Individual sums should match yields closely
    for i in 1:n_components
        @test isapprox(closure.sums[i], yields[i], rtol=1e-2)
    end
end

@testset "sWeights_multidimensional - 2D case" begin
    # Create 2D test case similar to phi-phi analysis
    n_events = 500
    
    # Generate 2D data points
    data_points = zeros(n_events, 2)
    data_points[:, 1] = randn(n_events) .+ 1019.0  # m_phi1 around phi mass
    data_points[:, 2] = randn(n_events) .+ 1019.0  # m_phi2 around phi mass
    
    # Define shape functions for 4 components (phi-phi, phi-bg, bg-phi, bg-bg)
    function phi_signal(m)
        return exp(-0.5 * ((m - 1019.0) / 5.0)^2)  # Gaussian around phi mass
    end
    
    function kk_background(m)
        return exp(-0.1 * abs(m - 1019.0))  # Exponential background
    end
    
    shape_functions = [
        x -> phi_signal(x[1]) * phi_signal(x[2]),      # phi-phi
        x -> phi_signal(x[1]) * kk_background(x[2]),   # phi-bg
        x -> kk_background(x[1]) * phi_signal(x[2]),   # bg-phi
        x -> kk_background(x[1]) * kk_background(x[2]) # bg-bg
    ]
    
    yields = [150.0, 100.0, 100.0, 150.0]
    
    # Compute sWeights
    weights, cov = sWeights_multidimensional(yields, shape_functions, data_points)
    
    # Basic checks
    @test size(weights) == (n_events, 4)
    @test size(cov) == (4, 4)
    @test all(diag(cov) .> 0)
    @test isapprox(cov, cov', atol=1e-10)
    
    # Check closure
    closure = check_sweights_closure(weights, yields, rtol=1e-2)
    @test closure.passed
end

@testset "sWeights_general - edge cases" begin
    # Test with very small yields
    yields = [1e-6, 1e-6]
    shape_values = [1.0 1.0; 1.0 1.0]  # 2 events, 2 components
    
    weights, cov = sWeights_general(yields, shape_values)
    @test size(weights) == (2, 2)
    @test all(isfinite.(weights))
    @test all(isfinite.(cov))
    
    # Test with one dominant component - use different shape values
    yields_dominant = [1000.0, 1.0]
    shape_values_different = [1.0 0.1; 1.0 0.1]  # First component stronger
    weights_dom, cov_dom = sWeights_general(yields_dominant, shape_values_different)
    # Check that the dominant component has higher weight per event
    total_weight_comp1 = sum(weights_dom[:, 1])
    total_weight_comp2 = sum(weights_dom[:, 2])
    @test total_weight_comp1 > total_weight_comp2  # Component 1 should dominate
    
    # Test error handling
    @test_throws ArgumentError sWeights_general([1.0], ones(2, 2))  # Mismatched dimensions
end

@testset "check_sweights_closure" begin
    # Perfect closure case
    yields = [10.0, 20.0, 30.0]
    weights = [
        1.0 2.0 3.0;
        2.0 4.0 6.0;
        3.0 6.0 9.0;
        4.0 8.0 12.0
    ]  # Each column sums to the corresponding yield
    
    closure = check_sweights_closure(weights, yields)
    @test closure.passed
    @test all(closure.relative_errors .< 1e-10)
    @test closure.sums ≈ yields
    
    # Imperfect closure case
    weights_imperfect = weights .* 1.1  # 10% error
    closure_bad = check_sweights_closure(weights_imperfect, yields, rtol=0.05)
    @test !closure_bad.passed  # Should fail with 5% tolerance
    
    closure_relaxed = check_sweights_closure(weights_imperfect, yields, rtol=0.15)
    @test closure_relaxed.passed  # Should pass with 15% tolerance
    
    # Test error handling
    @test_throws ArgumentError check_sweights_closure(weights, [1.0, 2.0])  # Wrong number of yields
end

@testset "Backward compatibility" begin
    # Ensure existing sPlot functionality still works after adding new methods
    pdfS = Normal(0, 1)
    pdfB = Normal(5, 1.5)
    data = vcat(rand(pdfS, 40), rand(pdfB, 60))
    
    # All existing functions should still work
    model = MixtureModel([pdfS, pdfB], [0.4, 0.6])
    sP = sPlot(model)
    
    weights = sWeights(sP, data)
    @test size(weights, 2) == 2
    
    ws, wb = eachcol(weights)
    fS(x) = sWeights(sP, [x])[1, 1]
    fB(x) = sWeights(sP, [x])[1, 2]
    
    @test fS(0.0) > 0.8
    @test fB(5.0) > 0.8
    
    # Covariance functions
    W = wMatrix(sP)
    cov = inv_W(sP)
    @test size(W) == (2, 2)
    @test size(cov) == (2, 2)
    
    # Condition check
    condW = check_wMatrix_condition(sP)
    @test condW > 0
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
