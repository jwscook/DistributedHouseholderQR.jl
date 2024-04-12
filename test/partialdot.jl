using Random, Test
using BenchmarkTools, LinearAlgebra

const np = try parse(Int, ARGS[1]); catch; 2; end

Random.seed!(0)

using DistributedHouseholderQR
const DHQR = DistributedHouseholderQR

@testset "Partialdot" begin
   for N in 1:20
    a = rand(ComplexF64, N)
    b = rand(ComplexF64, N)
    answers = Dict(i=>dot(a[i:end], b[i:end]) for i in eachindex(a, b))

    for i in eachindex(a, b)
      result = DHQR.partialdot(a, b, i:length(a), ComplexF64)
      @test result ≈ answers[i]
    end
  end
end
