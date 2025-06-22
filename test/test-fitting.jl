using Test
using HighEnergyTools
using Random

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
