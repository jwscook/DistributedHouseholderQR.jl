using Random, Distributed, Base.Threads

const np = try parse(Int, ARGS[1]); catch; 2; end

Random.seed!(0)

using DistributedHouseholderQR
addprocs(np, exeflags=["--proj=@.","-t $(nthreads())"])
@everywhere using Test
@everywhere begin
using LinearAlgebra, Random, Test
using Distributed, DistributedArrays, SharedArrays
using ThreadPinning
ThreadPinning.Prefs.set_os_warning(false)
@static if Sys.islinux()
  ThreadPinning.pinthreads(:cores)
end
using DistributedHouseholderQR
LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

const DHQR = DistributedHouseholderQR
end

@testset "Distributed Householder QR" begin
  for T in (ComplexF64, ), mn in ((3, 2),)#((11, 10), (550, 500), (1100, 1000), (2200, 2000),)# (4400, 4000))
    m, n = mn
    A = rand(T, m, n)
    b = rand(T, m)
    A1 = deepcopy(Matrix(A))
    b1 = deepcopy(Vector(b))
    tl = @elapsed x1 = LinearAlgebra.qr!(A1, NoPivot()) \ b1
    println("The stdlib took $tl seconds for m=$m and n=$n")
    A2 = deepcopy(Matrix(A))
    b2 = deepcopy(Vector(b))
    ta = @elapsed begin
      α2 = zeros(T, n)
#      H, α = DHQR.householder!(A2, α2)
#      x2 = DHQR.solve_householder!(b2, H, α)
      x2 = DHQR.qr!(A2) \ b2
    end
    # distribute A across columns
    A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
    α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
    b3 = SharedArray(deepcopy(b))
    tb = @elapsed  begin
      #H, α = DHQR.householder!(A3, α3)
      #x3 = DHQR.solve_householder!(b3, H, α)
      qrA = DHQR.qr!(A3)
      @show qrA
      x3 = qrA \ deepcopy(b)
    end
    A4 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (nworkers(), 1))
    α4 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
    b4 = SharedArray(deepcopy(b))
    tc = @elapsed  begin
      #H, α = DHQR.householder!(A4, α4)
      #x4 = DHQR.solve_householder!(b4, H, α)
      qrA = DHQR.qr!(A4)
      @show qrA
      x4 = qrA \ deepcopy(b)
    end
    @testset "stlib (threaded)" begin
      @test norm(A' * A * x1 .- A' * b) < sqrt(eps())
    end
    @testset "threaded only" begin
      @test norm(A' * A * x2 .- A' * b) < sqrt(eps())
    end
    @testset "column distribributed + threaded" begin
      @test norm(A' * A * x3 .- A' * b) < sqrt(eps())
    end
    @testset "row distribributed + threaded" begin
      @test norm(A' * A * x4 .- A' * b) < sqrt(eps())
    end
    println("The threaded undistributed version $(ta/tl) times longer")
    println("The threaded and column distributed version $(tb/tl) times longer")
    println("The threaded and row distributed version $(tc/tl) times longer")
  end
end


