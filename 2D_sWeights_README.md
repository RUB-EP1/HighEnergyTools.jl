# 2D sWeights Extension for HighEnergyTools.jl

This extension adds support for multi-dimensional sWeights to HighEnergyTools.jl, enabling statistical background subtraction for complex analyses like the X2VV ccbar->phi phi decay study.

## New Functions

### Core Functions

#### `sWeights_general(yields, shape_values)`
The most general implementation that works with arbitrary numbers of components and pre-computed shape values.

**Arguments:**
- `yields`: Vector of fitted component yields [N₁, N₂, ..., Nₖ]
- `shape_values`: Matrix where shape_values[i,j] = Pⱼ(xᵢ) is the j-th component's shape function value at the i-th data point

**Returns:**
- `(weights, covariance)`: sWeights matrix and covariance matrix of yields

#### `sWeights_multidimensional(yields, shape_functions, data_points)`
Convenient wrapper for multi-dimensional data using shape functions.

**Arguments:**
- `yields`: Vector of fitted component yields
- `shape_functions`: Vector of functions where shape_functions[i](x) evaluates the i-th component
- `data_points`: Matrix where each row is a data point (n_events × n_dimensions)

**Returns:**
- `(weights, covariance)`: sWeights matrix and covariance matrix

### Utility Functions

#### `check_sweights_closure(weights, yields; rtol=1e-3)`
Validates the closure property: Σₙ wᵢ(n) ≈ Nᵢ for each component i.

**Returns:**
- NamedTuple with `passed`, `relative_errors`, and `sums` fields

## Example Usage

### 2D φφ Analysis

```julia
using HighEnergyTools

# Define shape functions for four components
phi_signal(m) = exp(-0.5 * ((m - 1019.461) / 3.0)^2)
kk_background(m) = 1.0  # Flat background

shape_functions = [
    x -> phi_signal(x[1]) * phi_signal(x[2]),      # φφ
    x -> phi_signal(x[1]) * kk_background(x[2]),   # φ(bg)
    x -> kk_background(x[1]) * phi_signal(x[2]),   # (bg)φ
    x -> kk_background(x[1]) * kk_background(x[2]) # (bg)(bg)
]

# Fitted yields from your 2D mass fit
yields = [N_phiphi, N_phi1kk2, N_kk1phi2, N_kkkk]

# Data points: [m_phi1, m_phi2] for each event
data = [m_phi1 m_phi2]  # n_events × 2 matrix

# Compute sWeights
weights, cov = sWeights_multidimensional(yields, shape_functions, data)

# Check closure
closure = check_sweights_closure(weights, yields)
if closure.passed
    println("✓ Closure test passed")
end

# Extract component weights
phiphi_weights = weights[:, 1]
phi1kk2_weights = weights[:, 2] 
kk1phi2_weights = weights[:, 3]
kkkk_weights = weights[:, 4]
```

### Alternative: Pre-computed Shape Values

If you already have shape values computed (e.g., from a fit), you can use the more direct approach:

```julia
# shape_values[i,j] = value of component j's shape function at event i
shape_values = zeros(n_events, 4)
for i in 1:n_events
    # Evaluate your model components here
    shape_values[i, 1] = S1[i] * S2[i]     # φφ component
    shape_values[i, 2] = S1[i] * B2[i]     # φ(bg) component  
    shape_values[i, 3] = B1[i] * S2[i]     # (bg)φ component
    shape_values[i, 4] = B1[i] * B2[i]     # (bg)(bg) component
end

weights, cov = sWeights_general(yields, shape_values)
```

## Mathematical Background

The implementation follows the Pivk & Le Diberder method (NIM A555 (2005) 356):

- **V⁻¹ᵢⱼ = Σₙ Pᵢ(xₙ)Pⱼ(xₙ)/F(xₙ)²** where F(x) = Σₖ NₖPₖ(x)
- **wᵢ(n) = Σⱼ Vᵢⱼ Pⱼ(xₙ)/F(xₙ)**
- **Ensures Σₙ wᵢ(n) = Nᵢ** (closure condition)

## Backward Compatibility

All existing 1D sWeights functionality remains unchanged:
- `sPlot` objects and `sWeights` functions work exactly as before
- `wMatrix`, `inv_W`, `check_wMatrix_condition` continue to work
- All existing tests pass

## Migration from X2VV Analysis

To migrate from the X2VV-specific implementation:

```julia
# Old X2VV code:
include("src/sweight.jl")
using .SWeightsInternal
weights, V = compute_sweights(N_vec, P_cache)

# New HighEnergyTools.jl code:
using HighEnergyTools
weights, V = sWeights_general(N_vec, shape_values)
# where shape_values[i,j] = P_cache[i][j]
```

## Testing

Run the comprehensive test suite:
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The test suite includes:
- Multi-component 2D cases (4-component φφ analysis)
- Closure property validation
- Edge cases and error handling
- Backward compatibility verification

## Example Script

See `examples/phi_phi_2d_sweights_example.jl` for a complete demonstration of the 2D sWeights functionality applied to a synthetic φφ analysis dataset.