using Random, Distributed, Test

Random.seed!(0)

addprocs(2, exeflags="-t 1")

@everywhere begin
using LinearAlgebra, Random
using Distributed, DistributedArrays, SharedArrays
using DistributedHouseholderQR

LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

@testset "Distributed Householder QR" begin
    for T in (Float64, ComplexF64), mn in ((1200, 800), (2400, 2000),)# (4800, 4000))
#for T in (Float64, ComplexF64), mn in ((3, 2), (12, 10), (24, 20), (120, 100))
#for T in (Float64, ComplexF64), mn in ((2, 2), (3, 2), (120, 100),)
  m, n = mn
  @show T, m, n
  A = rand(T, m, n)
  b = rand(T, m)
  A1 = deepcopy(Matrix(A))
  b1 = deepcopy(Vector(b))
  t1 = @elapsed x1 = qr!(A1, NoPivot()) \ b1
  A2 = deepcopy(Matrix(A))
  b2 = deepcopy(Vector(b))
  ta = @elapsed begin
    α2 = zeros(T, n)
    H, α = householder!(deepcopy(A2), deepcopy(α2))
    x2 = solve_householder!(deepcopy(b2), H, α)
    @show "normal timings"
    @time H, α = householder!(A2, α2)
    @time x2 = solve_householder!(b2, H, α)
  end
  # distribute A across columns
  A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
  _A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
   α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
  _α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
   b3 = SharedArray(deepcopy(b))
  _b3 = SharedArray(deepcopy(b))
  tb = @elapsed  begin
    H, α = householder!(_A3, _α3)
    solve_householder!(_b3, H, α)
    @show "distrib timings"
    @time H, α = householder!(A3, α3)
    @time x3 = solve_householder!(b3, H, α)
  end
  @testset "serial, library" begin
    @test norm(A' * A * x1 .- A' * b) < sqrt(eps())
  end
  @testset "serial, this" begin
    @test norm(A' * A * x2 .- A' * b) < sqrt(eps())
  end
  @testset "distrib, this" begin
    @test norm(A' * A * x3 .- A' * b) < sqrt(eps())
  end
  @show ta / t1
  @show tb / t1
end


