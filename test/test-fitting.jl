using HighEnergyTools
using ComponentArrays
using Random
using Distributions
using FHist

@testset "chi2" begin
    h = Hist1D(rand(Normal(0, 1), 1000); binedges = range(start = -3, stop = 3, step = 0.5))
    d = Normal(0, 1)
    χ² = chi2(h, d)
    @test χ² ≥ 0
    #
    model(x) = pdf(d, x) * integral(h; width = true)
    χ²2 = chi2(h, model)
    @test isapprox(χ², χ²2; atol = 1e-8)
    # Test with zero-count bins
    h2 = Hist1D(
        vcat(randn(1000), fill(10.0, 10));
        binedges = range(start = -3, stop = 11, step = 0.5),
    )
    χ²3 = chi2(h2, d)
    @test χ²3 ≥ 0
end

@testset "Simple fitting" begin
    init_pars = (; μ = 0.35, σ = 0.8, a = 1.0)
    support = (-4.0, 4.0)
    Random.seed!(11122)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)
    ext_unbinned_fit = fit_enll(model, init_pars, data; support = support)
    best_pars_extnll = typeof(init_pars)(ext_unbinned_fit.minimizer)
    # @test ext_unbinned_fit.ls_success
    @test isapprox(best_pars_extnll.μ, 0.42965348; atol = 1e-6)
end


#
anka = Anka(1.1, 3.3)
init_pars = ComponentArray(sig = (μ = 2.2, σ = 0.06), bgd = (coeffs = [1.5, 1.1],), logfB = 0.0)
m = build_model(anka, init_pars)

Random.seed!(11122)
data = rand(m, 1000)

fit_res = fit_nll(anka, data, init_pars)
best_pars = fit_res.minimizer
#
@testset "Fitting NLL" begin
    @test isapprox(init_pars.logfB, best_pars.logfB; atol = 1e-1)
    @test isapprox(init_pars.sig.μ, best_pars.sig.μ; atol = 1e-2)
    @test isapprox(init_pars.sig.σ, best_pars.sig.σ; atol = 1e-3)
    @test sum(abs2, init_pars.bgd.coeffs - best_pars.bgd.coeffs) |> sqrt < 0.1
end

# # for visual inspection
# using Plots
# theme(:boxed)
# best_model = build_model(anka, best_pars)
# stephist(data; normalize = true, ylims = (0, :auto))
# plot!(x -> pdf(best_model, x), 1.1, 3.3, ylims = (0, :auto))
