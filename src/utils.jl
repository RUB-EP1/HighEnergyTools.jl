
"""
    interpolate_to_zero(two_x, two_y)

Interpolate to zero based on two points.

# Arguments
- `two_x::Vector`: A vector containing two x-values.
- `two_y::Vector`: A vector containing two y-values corresponding to `two_x`.

# Returns
- `Float64`: The interpolated x-value where y is zero.
"""
function interpolate_to_zero(two_x, two_y)
    w_left = 1 ./ two_y .* [1, -1]
    w_left ./= sum(w_left)
    return two_x' * w_left
end

"""
    find_zero_two_sides(xv, yv)

Find the zero crossings on both sides of the x-axis.

# Arguments
- `xv::AbstractVector`: A vector of x-values.
- `yv::AbstractVector`: A vector of y-values corresponding to `xv`.

# Returns
- `Vector{Float64}`: A vector containing two x-values where the y-values are zero.
"""
function find_zero_two_sides(xv, yv)
    yxv = yv .* xv
    _left = findfirst(x -> x > 0, yxv)
    _right = findlast(x -> x < 0, yxv)
    #
    x_left_zero = interpolate_to_zero([xv[_left-1], xv[_left]], [yv[_left-1], yv[_left]])
    x_right_zero =
        interpolate_to_zero([xv[_right], xv[_right+1]], [yv[_right], yv[_right+1]])
    #
    [x_left_zero, x_right_zero]
end

"""
    support_union(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution) -> Tuple{Float64, Float64}

Compute the union of the supports (ranges) of two `UnivariateDistribution` distributions.

This function determines the smallest interval that fully contains the support of both the signal and background distributions. It is useful for defining integration limits when combining or comparing two distributions, such as in sPlot or likelihood calculations.

# Arguments
- `pdfS::UnivariateDistribution`: The signal distribution (from Distributions.jl).
- `pdfB::UnivariateDistribution`: The background distribution.

# Returns
- `Tuple{Float64, Float64}`: A tuple `(lower, upper)` representing the minimum and maximum x-values that cover the support of both models.

# Notes
- The function uses `extrema` to find the minimum and maximum values for each distribution.
- For most standard distributions, this will return finite values, but for distributions with infinite support (like Normal), the result will be `(-Inf, Inf)`.
- If you want to restrict the support to a finite range (e.g., for numerical integration), you may need to override or post-process the result.

# Example
```julia
using Distributions
pdfS = Normal(0, 1)
pdfB = Normal(5, 1.5)
lims = support_union(pdfS, pdfB)
println(lims)  # Output: (-Inf, Inf)
```
"""
function support_union(pdfS::UnivariateDistribution, pdfB::UnivariateDistribution)
    s1, s2 = extrema(pdfS)
    b1, b2 = extrema(pdfB)
    return (min(s1, b1), max(s2, b2))
end
