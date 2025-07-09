using HighEnergyTools
using Distributions
using DistributionsHEP
using Test

@testset "Chebyshev" begin
    a = HighEnergyTools.scaled_chebyshev([1, 1], (-1, 1))
    @test pdf(a, -1) ≈ 0.0
    @test pdf(a, 1) ≈ 1.0

    b = HighEnergyTools.scaled_chebyshev([1, 1], (3.0, 7.0))
    @test pdf(b, 3.0) ≈ 0.0
    @test pdf(b, 7.0) ≈ 0.5
end

anka = Anka(1.1, 3.3)
pars = (sig = (μ = 2.2, σ = 0.06), bgd = (coeffs = [1.5, 1.1],), logfB = 0.0)
model = build_model(anka, pars)

# # for visual inspection
# using Plots
# theme(:boxed)
# let
#     plot(x -> pdf(model, x), 1.1, 3.3, ylims = (0, :auto))
#     plot!(x -> pdf(model.components[1], x) * model.prior.p[1], 1.1, 3.3, fill = 0)
#     plot!(x -> pdf(model.components[2], x) * model.prior.p[2], 1.1, 3.3, ls=:dash)
# end

@testset "Anka model" begin
    @test model.components[1] isa Truncated{<:Normal}
    @test model.components[2] isa
          LocationScale{A,B,<:DistributionsHEP.StandardChebyshev} where {A,B}
    @test model.prior.p[1] ≈ 0.5
    @test model.prior.p[2] ≈ 0.5
end


frida = Frida(1.1, 3.3)
pars = (
    sig1 = (μ = 2.1, σ = 0.05),
    sig2 = (μ = 2.4, σ = 0.08),
    bgd = (coeffs = [1.5, 1.1],),
    logfS1 = -1.0,
    logfS2 = -1.0,
)
model = build_model(frida, pars)

# # for visual inspection
# using Plots
# theme(:boxed)
# let
#     plot(x -> pdf(model, x), 1.1, 3.3, ylims = (0, :auto))
#     plot!(x -> pdf(model.components[1], x) * model.prior.p[1], 1.1, 3.3, fill = 0)
#     plot!(x -> pdf(model.components[2], x) * model.prior.p[2], 1.1, 3.3, fill = 0)
#     plot!(x -> pdf(model.components[3], x) * model.prior.p[3], 1.1, 3.3)
# end

@testset "Frida model" begin
    @test model.components[1] isa Truncated{<:Normal}
    @test model.components[2] isa Truncated{<:Normal}
    @test model.components[3] isa
          LocationScale{A,B,<:DistributionsHEP.StandardChebyshev} where {A,B}
end
