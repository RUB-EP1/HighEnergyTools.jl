using Test
using HighEnergyTools
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
    @test best_pars_extnll.μ ≈ 0.4296536896499441
end
