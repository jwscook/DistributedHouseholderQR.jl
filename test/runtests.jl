using Random, Distributed, Base.Threads
using BenchmarkTools

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
splits(np, N, p) = round(Int, (N / sqrt(np)) * sqrt(p))
lorange(np, N, p) = max(1, splits(np, N, p-1) + 1)
hirange(np, N, p) = min(N, splits(np, N, p))
end

@testset "Distributed Householder QR" begin
  for T in (ComplexF64, ), mn in ((11, 10), (550, 500), (1100, 1000), (2200, 2000), (4400, 4000),)# (8800, 8000))
    m, n = mn
    A = rand(T, m, n)
    b = rand(T, m)
    A1 = deepcopy(Matrix(A))
    b1 = deepcopy(Vector(b))
    x1 = LinearAlgebra.qr!(A1, NoPivot()) \ b1
    bm1 =@benchmark LinearAlgebra.qr!($(deepcopy(A))) \ $(deepcopy(b))
    tl = minimum(bm1).time / 1e9
    println("The stdlib took $tl seconds for m=$m and n=$n")
    A2 = deepcopy(Matrix(A))
    b2 = deepcopy(Vector(b))
    _A2 = deepcopy(Matrix(A))
    _b2 = deepcopy(Vector(b))
    x2 = DHQR.qr!(A2) \ b2
    bm2 = @benchmark DHQR.qr!($(deepcopy(_A2))) \ $(_b2)
    ta = minimum(bm2).time / 1e9
    # distribute A across columns
    A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
    #ras = [@spawnat workers()[i] A[:, lorange(np, n, i):hirange(np, n, i)] for i in 1:np]
    #A3 = DArray(reshape(ras, (1, nworkers())))

    α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
    b3 = SharedArray(deepcopy(b))
    _A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
    _b3 = deepcopy(b)
    qrA = DHQR.qr!(A3)
    x3 = qrA \ deepcopy(b)
    bm3 = @benchmark DHQR.qr!($(_A3)) \ $(_b3)
    tb = minimum(bm3.times) / 1e9

    @testset "stlib (threaded)" begin
      @test norm(A' * A * x1 .- A' * b) < sqrt(eps())
    end
    @testset "this threaded only" begin
      @test norm(A' * A * x2 .- A' * b) < sqrt(eps())
    end
    @testset "this distribributed + threaded" begin
      @test norm(A' * A * x3 .- A' * b) < sqrt(eps())
    end
    println("The threaded undistributed version $(ta/tl) times longer")
    println("The threaded and distributed version $(tb/tl) times longer")
  end
end


