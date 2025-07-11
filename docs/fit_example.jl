### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 0be21b32-5cf8-11f0-1755-81024a857a76
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    #
    using Plots
    using HighEnergyTools.DistributionsHEP
    using HighEnergyTools.Parameters
    using HighEnergyTools.Optim
    using HighEnergyTools.FHist
    using HighEnergyTools
    using ComponentArrays
    using Plots
end

# ╔═╡ afe3198b-087f-45db-b7fe-d5ef2cfc2169
theme(:boxed)

# ╔═╡ 3323dca7-d54b-4a94-94fa-405a7283debb
anka = Anka(3, 5)

# ╔═╡ 350c3cb4-1549-4171-ba00-d3da19b85f5a
pars = (sig = (μ = 4.2, σ = 0.06), bgd = (coeffs = [1.5, -1.1],), logfB = 1.0)

# ╔═╡ 04c48759-114b-4e6f-ad0a-93d6016c4a60
data = let
    _data = vcat(1 .- sqrt.(rand(3000)), rand(1000), 0.4 .+ 0.1 .* randn(10000))
    filter(_data) do x
        0 < x < 1
    end
    _data .* 2 .+ 3
end;

# ╔═╡ cfec9b34-8c51-4235-ae65-7c45f893e71c
fit_res = fit_nll(data, ComponentArray(pars)) do p
    build_model(anka, p)
end

# ╔═╡ 7ba209a4-afc8-46e8-9c9d-e08dd5f2b59d
let
    binedges = range(extrema(data)..., 100)
    model = build_model(anka, ComponentArray(pars))
    δx = binedges[2] - binedges[1]
    scale = length(data) * δx
    #
    stephist(data, bins = binedges)
    plot!(x -> pdf(model, x) * scale, extrema(data)...)
    #
    best_model = build_model(anka, fit_res.minimizer)
    plot!(x -> pdf(best_model, x) * scale, extrema(data)...)
end

# ╔═╡ Cell order:
# ╠═0be21b32-5cf8-11f0-1755-81024a857a76
# ╠═afe3198b-087f-45db-b7fe-d5ef2cfc2169
# ╠═3323dca7-d54b-4a94-94fa-405a7283debb
# ╠═350c3cb4-1549-4171-ba00-d3da19b85f5a
# ╠═04c48759-114b-4e6f-ad0a-93d6016c4a60
# ╠═cfec9b34-8c51-4235-ae65-7c45f893e71c
# ╠═7ba209a4-afc8-46e8-9c9d-e08dd5f2b59d
