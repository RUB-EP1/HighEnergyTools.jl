#!/usr/bin/env julia

"""
Example: 2D sWeights for φφ Analysis using HighEnergyTools.jl

This example demonstrates how to use the new 2D sWeights functionality in HighEnergyTools.jl
for a four-component analysis similar to the X2VV ccbar->phi phi analysis.

The four components are:
1. φφ: both φ mesons are true signal 
2. φ(background): first φ is signal, second is combinatorial background
3. (background)φ: first φ is background, second is signal  
4. (background)(background): both are combinatorial background

Author: X2VV Analysis Team
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HighEnergyTools
using Distributions
using Random
using Plots
using LinearAlgebra
using Statistics

# Constants for phi meson
const PHI_MASS = 1019.461  # MeV/c²
const PHI_WIDTH = 4.249    # MeV/c²
const K_MASS = 493.677     # MeV/c²

"""
Generate synthetic 2D phi-phi data for demonstration
"""
function generate_synthetic_phi_phi_data(n_events::Int=1000)
    Random.seed!(42)  # For reproducibility
    
    # Component yields
    n_phiphi = 300
    n_phi1bg2 = 200 
    n_bg1phi2 = 200
    n_bgbg = 300
    
    data_points = zeros(n_events, 2)
    true_labels = zeros(Int, n_events)
    
    idx = 1
    
    # Component 1: φφ (both signal)
    for i in 1:n_phiphi
        data_points[idx, 1] = rand(Normal(PHI_MASS, 3.0))
        data_points[idx, 2] = rand(Normal(PHI_MASS, 3.0)) 
        true_labels[idx] = 1
        idx += 1
    end
    
    # Component 2: φ(bg) (first signal, second background)
    for i in 1:n_phi1bg2
        data_points[idx, 1] = rand(Normal(PHI_MASS, 3.0))
        data_points[idx, 2] = rand(Uniform(1000.0, 1040.0))  # Flat background
        true_labels[idx] = 2
        idx += 1
    end
    
    # Component 3: (bg)φ (first background, second signal)
    for i in 1:n_bg1phi2
        data_points[idx, 1] = rand(Uniform(1000.0, 1040.0))  # Flat background
        data_points[idx, 2] = rand(Normal(PHI_MASS, 3.0))
        true_labels[idx] = 3
        idx += 1
    end
    
    # Component 4: (bg)(bg) (both background)
    for i in 1:n_bgbg
        data_points[idx, 1] = rand(Uniform(1000.0, 1040.0))
        data_points[idx, 2] = rand(Uniform(1000.0, 1040.0))
        true_labels[idx] = 4
        idx += 1
    end
    
    return data_points, true_labels, [n_phiphi, n_phi1bg2, n_bg1phi2, n_bgbg]
end

"""
Define shape functions for the four-component model
"""
function define_shape_functions()
    # Phi signal shape (simplified Gaussian)
    function phi_signal(m)
        return exp(-0.5 * ((m - PHI_MASS) / 3.0)^2)
    end
    
    # Background shape (uniform-like)
    function kk_background(m)
        return 1.0  # Flat background
    end
    
    # Four-component shape functions
    shape_functions = [
        x -> phi_signal(x[1]) * phi_signal(x[2]),      # φφ
        x -> phi_signal(x[1]) * kk_background(x[2]),   # φ(bg)
        x -> kk_background(x[1]) * phi_signal(x[2]),   # (bg)φ
        x -> kk_background(x[1]) * kk_background(x[2]) # (bg)(bg)
    ]
    
    return shape_functions
end

"""
Main demonstration function
"""
function demo_2d_sweights()
    println("=" ^ 60)
    println("2D sWeights Demonstration for φφ Analysis")
    println("=" ^ 60)
    
    # Generate synthetic data
    println("1. Generating synthetic φφ data...")
    data_points, true_labels, true_yields = generate_synthetic_phi_phi_data(1000)
    n_events, n_dims = size(data_points)
    
    println("   Generated $n_events events in $n_dims dimensions")
    println("   True yields: φφ=$(true_yields[1]), φ(bg)=$(true_yields[2]), (bg)φ=$(true_yields[3]), (bg)(bg)=$(true_yields[4])")
    
    # Define shape functions
    println("\n2. Defining shape functions...")
    shape_functions = define_shape_functions()
    println("   Defined $(length(shape_functions)) component shape functions")
    
    # Compute sWeights using the general multidimensional method
    println("\n3. Computing 2D sWeights...")
    fitted_yields = [320.0, 180.0, 220.0, 280.0]  # Simulated fitted yields (with some error)
    
    weights, covariance = sWeights_multidimensional(fitted_yields, shape_functions, data_points)
    
    println("   Computed sWeights matrix: $(size(weights))")
    println("   Covariance matrix: $(size(covariance))")
    
    # Check closure property
    println("\n4. Checking closure property...")
    closure = check_sweights_closure(weights, fitted_yields, rtol=0.05)
    
    if closure.passed
        println("   ✓ Closure test PASSED")
    else
        println("   ✗ Closure test FAILED")
    end
    
    for i in 1:length(fitted_yields)
        rel_err_pct = closure.relative_errors[i] * 100
        println("   Component $i: fitted=$(fitted_yields[i]), sum_weights=$(round(closure.sums[i], digits=1)), rel_error=$(round(rel_err_pct, digits=2))%")
    end
    
    # Alternative: Using the general method with pre-computed shape values
    println("\n5. Alternative: Using sWeights_general with pre-computed shape values...")
    
    # Pre-compute shape values matrix
    shape_values = zeros(n_events, length(shape_functions))
    for i in 1:n_events
        for j in 1:length(shape_functions)
            shape_values[i, j] = shape_functions[j](data_points[i, :])
        end
    end
    
    weights2, cov2 = sWeights_general(fitted_yields, shape_values)
    
    # Verify both methods give the same result
    @assert weights ≈ weights2 "Methods should give identical results"
    @assert covariance ≈ cov2 "Covariance matrices should be identical"
    println("   ✓ Both methods give identical results")
    
    # Analyze weight properties
    println("\n6. Analyzing sWeight properties...")
    
    # Component weights by true component
    component_names = ["φφ", "φ(bg)", "(bg)φ", "(bg)(bg)"]
    
    for true_comp in 1:4
        mask = true_labels .== true_comp
        if sum(mask) > 0
            avg_weights = [mean(weights[mask, i]) for i in 1:4]
            println("   True $( component_names[true_comp]) events:")
            for i in 1:4
                println("     Average weight for $(component_names[i]): $(round(avg_weights[i], digits=3))")
            end
            # The weight for the true component should be highest
            max_weight_comp = argmax(avg_weights)
            if max_weight_comp == true_comp
                println("     ✓ Highest weight correctly assigned to true component")
            else
                println("     ⚠ Highest weight assigned to $(component_names[max_weight_comp]) instead of $(component_names[true_comp])")
            end
        end
    end
    
    # Check weight ranges
    println("\n7. Weight statistics:")
    for i in 1:4
        w_min, w_max = extrema(weights[:, i])
        w_mean = mean(weights[:, i])
        println("   $(component_names[i]): range=[$(round(w_min, digits=3)), $(round(w_max, digits=3))], mean=$(round(w_mean, digits=3))")
    end
    
    # Covariance matrix properties
    println("\n8. Covariance matrix properties:")
    eigenvals = eigvals(covariance)
    cond_number = maximum(eigenvals) / minimum(eigenvals)
    println("   Condition number: $(round(cond_number, digits=1))")
    println("   All eigenvalues positive: $(all(eigenvals .> 0))")
    println("   Matrix symmetric: $(issymmetric(covariance))")
    
    println("\n" * "=" ^ 60)
    println("2D sWeights demonstration completed successfully!")
    println("The new HighEnergyTools.jl functionality is ready for your φφ analysis.")
    println("=" ^ 60)
    
    return weights, covariance, data_points, true_labels
end

# Run the demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    demo_2d_sweights()
end