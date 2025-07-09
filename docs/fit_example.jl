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
    using HighEnergyTools.Distributions
    using HighEnergyTools.Parameters
    using HighEnergyTools.Optim
    using HighEnergyTools.FHist
    using HighEnergyTools
    using ComponentArrays
    using Plots
end

# ╔═╡ afe3198b-087f-45db-b7fe-d5ef2cfc2169
theme(:boxed)

# ╔═╡ 350c3cb4-1549-4171-ba00-d3da19b85f5a
pars = (sig = (μ = 4.2, σ = 0.06), bgd = (coeffs = [1.5, -1.1],), logfB = 1.0)

# ╔═╡ 04c48759-114b-4e6f-ad0a-93d6016c4a60
data = let
    _data = vcat(1 .- sqrt.(rand(8000)), rand(1000), 0.4 .+ 0.08 .* randn(10000))
    filter(_data) do x
        0 < x < 1
    end
    _data .* 2 .+ 3
end;

# ╔═╡ a0e8c5f2-2185-4f25-ae96-79bdaa299d3f
h0 = Hist1D(data, binedges=range(3, 5, 100))

# ╔═╡ f2f0bef9-fea4-4eb9-9507-6338231dae61
md"""
## Regular nll fit
"""

# ╔═╡ 3323dca7-d54b-4a94-94fa-405a7283debb
anka = Anka(3, 5)

# ╔═╡ 29f7e4c4-0b32-450f-9643-d234f7c30da5
md"""
## Extended nll fit

$μ^n  e^{-μ} / n!$

"""

# ╔═╡ 7cfa3ba8-174b-4ca8-975b-9dbffce85c85
begin
	struct Extended{M, T}
		model::M
		n::T
	end
	function HighEnergyTools.build_model(components::Vector, pars)
		@unpack yields = pars
		n = sum(yields)
		fractions = yields ./  n
		_model = MixtureModel(components, fractions)
		Extended(_model, n)
	end
	Distributions.pdf(p::Extended, x) = pdf(p.model, x)
	# 
	function HighEnergyTools.nll(p::Extended, data)
		_nll = HighEnergyTools.nll(p.model, data)
		n = length(data)
		μ = p.n
		return _nll + μ - n*log(μ)
	end
end

# ╔═╡ 18f485f6-9e66-4a8b-8265-ffbff90527ab
ext_pars = (;
			sig = (μ = 4.2, σ = 0.06),
			bgd = (coeffs = [1.5, -1.1],),
			yields=[5000.0, 5000.0])

# ╔═╡ 323b1e8b-5eab-4b52-8198-9b6bd757e0c4
begin
	const ExtendedAnka = Extended{<:Anka,T} where T
	function HighEnergyTools.build_model(e::ExtendedAnka, p)
		@unpack sig, bgd = p
		_model = build_model(anka, (; logfB = 0.0, sig, bgd))
	    build_model(_model.components, p)
	end
end

# ╔═╡ cfec9b34-8c51-4235-ae65-7c45f893e71c
fit_res = fit_nll(data, ComponentArray(pars)) do p
    build_model(anka, p)
end

# ╔═╡ 9a394c36-52dc-41fc-83a8-48f14283360b
# begin
# 	const ExtendedAnka = Extended{<:Anka,T} where T
# 	function HighEnergyTools.build_model(e::ExtendedAnka, p)
# 		@unpack sig, bgd = p
# 		components = [
# 		    truncated(Normal(sig.μ, sig.σ), anka.support...),
# 		    HighEnergyTools.scaled_chebyshev(bgd.coeffs, anka.support)
# 		]
# 	    build_model(components, p)
# 	end
# end

# ╔═╡ c0858d55-e640-46f2-9b42-def4475ae606
ext_fit_res = fit_nll(data, ComponentArray(ext_pars)) do p
    build_model(Extended(anka, 1.0), p)
end

# ╔═╡ 7ba209a4-afc8-46e8-9c9d-e08dd5f2b59d
let
    binedges = range(extrema(data)..., 100)
	h0 = Hist1D(data; binedges)
	# 
    model = build_model(anka, ComponentArray(pars))
    δx = binedges[2] - binedges[1]
    scale = length(data) * δx
    #
	plot()
    plot!(x -> pdf(model, x) * scale, extrema(data)..., lab="starting", lc=8, ls=:dash)
    #
    best_model = build_model(anka, fit_res.minimizer)
    plot!(x -> pdf(best_model, x) * scale, extrema(data)..., lw=2, lab="nll")
	# 
	best_ext_model = build_model(Extended(anka, 1.0), ext_fit_res.minimizer)
    plot!(x -> pdf(best_ext_model, x) * scale, extrema(data)..., lw=2, lab="extended")
	# 
    scatter!(bincenters(h0), bincounts(h0), yerr=sqrt.(bincounts(h0)), m=(3, :o, :black))
end

# ╔═╡ Cell order:
# ╠═0be21b32-5cf8-11f0-1755-81024a857a76
# ╠═afe3198b-087f-45db-b7fe-d5ef2cfc2169
# ╠═350c3cb4-1549-4171-ba00-d3da19b85f5a
# ╠═04c48759-114b-4e6f-ad0a-93d6016c4a60
# ╠═a0e8c5f2-2185-4f25-ae96-79bdaa299d3f
# ╟─f2f0bef9-fea4-4eb9-9507-6338231dae61
# ╠═3323dca7-d54b-4a94-94fa-405a7283debb
# ╠═cfec9b34-8c51-4235-ae65-7c45f893e71c
# ╟─29f7e4c4-0b32-450f-9643-d234f7c30da5
# ╠═7cfa3ba8-174b-4ca8-975b-9dbffce85c85
# ╠═18f485f6-9e66-4a8b-8265-ffbff90527ab
# ╠═323b1e8b-5eab-4b52-8198-9b6bd757e0c4
# ╠═9a394c36-52dc-41fc-83a8-48f14283360b
# ╠═c0858d55-e640-46f2-9b42-def4475ae606
# ╠═7ba209a4-afc8-46e8-9c9d-e08dd5f2b59d
