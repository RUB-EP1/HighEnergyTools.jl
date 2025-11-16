using HighEnergyTools
using HighEnergyTools.Distributions
using HighEnergyTools.FHist
using Plots

theme(:boxed)

h0 = Hist1D(randn(1000); binedges = -1.1:0.1:1.5)
model = truncated(Normal(0.0, 1.0), extrema(binedges(h0))...)

WD = WithData(h0)

# 1) just scaling curve
dx = (binedges(h0)[2] - binedges(h0)[1]) / 2
p1 = let
    plot(title = "WD test")
    plot!(sp = 1, x -> pdf(model, x), WD, lw = 2, lc = :cornflowerblue)
    scatter!(sp = 1, bincenters(h0), h0.bincounts, yerror = binerrors(h0), xerror = dx)
end

# pulls
mismatches = h0.bincounts - pdf.(Ref(model), bincenters(h0)) .* WD.factor
h_pull = Hist1D(; binedges = binedges(h0), bincounts = mismatches ./ binerrors(h0))

# 2) with pulls
p2 = let
    plot(layout = grid(2, 1, heights = (0.8, 0.2)), link = :x)
    plot!(sp=1, title = "curvehistpulls equivalent")
    plot!(sp = 1, x -> pdf(model, x), WD, lw = 2, lc = :cornflowerblue)
    scatter!(sp = 1, bincenters(h0), h0.bincounts, yerror = binerrors(h0), xerror = dx)
    scatter!(sp = 2, bincenters(h_pull), h_pull.bincounts, yerror = 1, xerror = dx)
end

# 3) shortcut
p3 = let
    curvehistpulls(x -> pdf(model, x), h0, xlab = "X-axis", ylab = "Y-axis")
    plot!(sp=1, title = "curvehistpulls shortcut")
end

# 4) curvehistpulls with data_scale_curve=false for models normalized to total events
data = randn(1000)
h0_doc = Hist1D(data; binedges=-1.5:0.1:2.5)
model_doc(x) = length(data) * exp(-x^2 / 2) / sqrt(2π)
# Use data_scale_curve = false for models normalized to total events
p4 = let
    curvehistpulls(model_doc, h0_doc, xlab="X-axis", ylab="Y-axis", data_scale_curve=false, title=["data_scale_curve=false" ""])
end

println("\nAll plotting examples completed. Displaying combined plot...")
plot(p1, p2, p3, p4, size=(1000, 1000), layout=grid(2, 2))
