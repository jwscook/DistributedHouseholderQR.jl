using Random, Distributed

Random.seed!(0)

using DistributedHouseholderQR
const DQHR = DistributedHouseholderQR
addprocs(2, exeflags=["--proj=@.","-t 10"])
@everywhere using Test
@everywhere begin
using LinearAlgebra, Random, Test
using Distributed, DistributedArrays, SharedArrays
using ThreadPinning
@static if Sys.islinux()
  ThreadPinning.pinthreads(:cores)
end
using DistributedHouseholderQR
LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

const DHQR = DistributedHouseholderQR
end

@testset "Distributed Householder QR" begin
  for T in (ComplexF64, ), mn in ((11, 10), (1100, 1000), (2200, 2000), (4400, 4000))
    m, n = mn
    A = rand(T, m, n)
    b = rand(T, m)
    A1 = deepcopy(Matrix(A))
    b1 = deepcopy(Vector(b))
    t1 = @elapsed x1 = qr!(A1, NoPivot()) \ b1
    A2 = deepcopy(Matrix(A))
    b2 = deepcopy(Vector(b))
    ta = @elapsed begin
      α2 = zeros(T, n)
      H, α = DHQR.householder!(A2, α2)
      x2 = DHQR.solve_householder!(b2, H, α)
    end
    # distribute A across columns
    A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
    α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
    b3 = SharedArray(deepcopy(b))
    tb = @elapsed  begin
      #H, α = DHQR.householder!(A3, α3)
      #x3 = DHQR.solve_householder!(b3, H, α)
      qrA = DHQR.qr!(A3)
      x3 = qrA \ deepcopy(b)
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
    @show t1, ta, tb, ta / t1, tb / t1
  end
end


