using HighEnergyTools.Distributions
using HighEnergyTools
using ComponentArrays
using Parameters
using Random
using Test

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
init_pars =
    ComponentArray(sig = (μ = 2.2, σ = 0.06), bgd = (coeffs = [1.5, 1.1],), logfB = 0.0)
m = build_model(anka, init_pars)

Random.seed!(11122)
data = rand(m, 1000)

fit_res = fit_nll(data, init_pars) do p
    build_model(anka, p)
end
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

@testset "Extended NLL fitting" begin
    # Test 1: Simple extended model with two Gaussians
    Random.seed!(12345)

    # Create components and yields
    components = [Normal(0.0, 1.0), Normal(2.0, 0.5)]
    yields = [100.0, 50.0]
    ext_model = Extended(components, (; yields))

    # Generate data from the first component
    data = rand(Normal(0.0, 1.0), 80)

    # Test PDF evaluation
    @test pdf(ext_model, 0.0) ≈ pdf(ext_model.model, 0.0)
    @test pdf(ext_model, 2.0) ≈ pdf(ext_model.model, 2.0)

    # Test NLL calculation
    nll_value = nll(ext_model, data)
    @test isfinite(nll_value)

    # Test 2: Extended fitting with Anka model
    anka = Anka(1.1, 3.3)
    init_pars = ComponentArray(
        sig = (μ = 2.2, σ = 0.06),
        bgd = (coeffs = [1.5, 1.1],),
        yields = [500.0, 500.0],
    )

    # Generate data from the model
    m = build_model(anka, (; init_pars.sig, init_pars.bgd, logfB = 0.0))
    data_ext = rand(m, 1000)

    # Fit using extended NLL
    fit_res_ext = fit_nll(data_ext, init_pars) do p
        @unpack sig, bgd = p
        _model = build_model(anka, (; logfB = 0.0, sig, bgd))
        Extended(_model.components, p)
    end
    best_pars_ext = fit_res_ext.minimizer

    # Test that yields are properly fitted
    @test isapprox(best_pars_ext.yields[1], 500.0, atol = 50.0)  # Rough estimate
    @test isapprox(best_pars_ext.yields[2], 500.0, atol = 50.0)  # Rough estimate
    @test isapprox(sum(best_pars_ext.yields), length(data_ext), atol = 1e-3)

    # Test 3: Extended model construction
    ext_model_anka = Extended(
        build_model(anka, (; logfB = 0.0, init_pars.sig, init_pars.bgd)).components, init_pars)
    @test ext_model_anka.n == sum(init_pars.yields)
    @test pdf(ext_model_anka, 2.0) > 0

    # Test 4: NLL comparison between regular and extended
    regular_nll = nll(m, data_ext)
    extended_nll = nll(ext_model_anka, data_ext)

    # Extended NLL should be different from regular NLL
    @test extended_nll != regular_nll
    @test extended_nll < regular_nll  # Extended NLL includes Poisson term

    # Test 5: Extended model with different yields
    yields_small = [100.0, 50.0]
    ext_model_small = Extended(components, (yields = yields_small,))
    yields_large = [1000.0, 500.0]
    ext_model_large = Extended(components, (yields = yields_large,))

    # NLL should be different for different yields
    nll_small = nll(ext_model_small, data)
    nll_large = nll(ext_model_large, data)
    @test nll_small != nll_large
end
