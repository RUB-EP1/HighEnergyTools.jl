using HighEnergyTools
using HighEnergyTools.Distributions
using Plots

theme(:boxed)

h0 = Hist1D(randn(1000); binedges = -1.1:0.1:1.5)
model = truncated(Normal(0.0, 1.0), extrema(binedges(h0))...)

WD = WithData(h0)

# 1) just scaling curve
let
    plot(title = "First plot: WD test")
    plot!(sp = 1, x -> pdf(model, x), WD, lw = 2, lc = :cornflowerblue)
    scatter!(sp = 1, bincenters(h0), h0.bincounts, yerror = binerrors(h0), xerror = dx)
end

# pulls
mismatches = h0.bincounts - pdf.(Ref(model), bincenters(h0)) .* WD.factor
h_pull = Hist1D(; binedges = binedges(h0), bincounts = mismatches ./ binerrors(h0))
dx = (binedges(h0)[2] - binedges(h0)[1]) / 2

# 2) with pulls
let
    plot(title = "Second plot: curvehistpulls equivalent")
    plot!(layout = grid(2, 1, heights = (0.8, 0.2)), link = :x)
    plot!(sp = 1, x -> pdf(model, x), WD, lw = 2, lc = :cornflowerblue)
    scatter!(sp = 1, bincenters(h0), h0.bincounts, yerror = binerrors(h0), xerror = dx)
    scatter!(sp = 2, bincenters(h_pull), h_pull.bincounts, yerror = 1, xerror = dx)
end

# 3) shortcut
let
    plot(title = "Third plot: curvehistpulls shortcut")
    curvehistpulls(x -> pdf(model, x), h0, xlab = "X-axis", ylab = "Y-axis")
end
