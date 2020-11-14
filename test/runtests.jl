using SequentialPointProcesses
using Distributions: logpdf
using Test

# Test that things run
@testset "SequentialPointProcesses.jl" begin
    M1(theta) = Softcore(d -> (2.3/d)^(theta))
    pp = rand(M1(4.5), (x=(0,10), y=(0,10)), 10)
    @test length(pp) == 10
    l1 = logpdf(M1(4.5), pp, 100)
    l2 = logpdf(M1(4.5), pp, SequentialPointProcesses.Options(nx=100, parallel=:threads))
    @test isfinite(l1)
    @test l1 ≈ l2 rtol=10eps(l1)

    M2(theta, prob2) = Mixture(M1(theta), SequentialPointProcesses.Uniform(), prob2)
    pp = rand(M2(4.5, 0.1), (x=(0,10), y=(0,10)), 10)
    @test length(pp) == 10
    l1 = logpdf(M2(4.5, 0.1), pp, 100)
    l2 = logpdf(M2(4.5, 0.1), pp, SequentialPointProcesses.Options(nx=100, parallel=:threads))
    @test isfinite(l1)
    @test l1 ≈ l2 rtol=10eps(l1)
end
