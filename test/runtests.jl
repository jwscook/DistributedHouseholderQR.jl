using Random, Distributed, Base.Threads
using BenchmarkTools

const np = try parse(Int, ARGS[1]); catch; 2; end

Random.seed!(0)

using DistributedHouseholderQR
addprocs(np, exeflags=["--proj=@.","-t $(nthreads())"])
@show nthreads()
@everywhere using Test
@everywhere begin
  using LinearAlgebra, Random, Test
  using Distributed, DistributedArrays, SharedArrays

  using ThreadPinning
  ThreadPinning.Prefs.set_os_warning(false)
  @static if Sys.islinux()
    nt = Threads.nthreads()
    p = max(1, myid()-1)
    cpus = if p > nworkers() / 2
      ThreadPinning.cpuids_per_numa()[2][(p-1)*nt+1-10:p*nt-10]
    else
      ThreadPinning.cpuids_per_numa()[1][(p-1)*nt+1:p*nt]
    end
    ThreadPinning.pinthreads(cpus; force=true)
    wait(@spawnat 1 ThreadPinning.pinthreads(1:nt; force=true))
    println("\tCPUs: ", getcpuids())
  end
  LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

  using DistributedHouseholderQR
  const DHQR = DistributedHouseholderQR

  #splits(np, N, p) = round(Int, (N / sqrt(np)) * sqrt(p))
  splits(np, N, p) = round(Int, N * (1 -sqrt((np-p) / np)))
  lorange(np, N, p) = max(1, splits(np, N, p-1) + 1)
  hirange(np, N, p) = min(N, splits(np, N, p))
end
using StatProfilerHTML, Profile
@testset "Distributed Householder QR" begin
  for mn in ((1100, 1000), (2200, 2000), (4400, 4000),),
      T in (Float32, ComplexF32, Float64, ComplexF64, )
    m, n = mn
    A = rand(T, m, n)
    b = rand(T, m)
    A1 = deepcopy(Matrix(A))
    b1 = deepcopy(Vector(b))
    x1 = LinearAlgebra.qr!(A1, NoPivot()) \ b1

    stdliberr = norm(A' * A * x1 .- A' * b)

    bm1 = @benchmark LinearAlgebra.qr!($(deepcopy(A))) \ $(deepcopy(b))
    tl = minimum(bm1).time / 1e9
    A2 = deepcopy(Matrix(A))
    b2 = deepcopy(Vector(b))
    _A2 = deepcopy(Matrix(A))
    _b2 = deepcopy(Vector(b))
    x2 = DHQR.qr!(A2) \ b2

    @testset "$T threaded only" begin
      @test norm(A' * A * x2 .- A' * b) < 8stdliberr
    end
    Profile.clear()
    @profilehtml DHQR.qr!(A2) \ b2
    bm2 = @benchmark DHQR.qr!($(deepcopy(_A2))) \ $(_b2)
    ta = minimum(bm2).time / 1e9
    # distribute A across columns
    #ras = [@spawnat workers()[i] A[:, lorange(np, n, i):hirange(np, n, i)] for i in 1:np]
    #A3 = DArray(reshape(ras, (1, nworkers())))
    A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))

    α3 = SharedArray(zeros(T,n))#zeros(T,n)#distribute(zeros(T, n))#
    b3 = SharedArray(deepcopy(b))
    _A3 = DArray(ij->A[ij[1], ij[2]], size(A), workers(), (1, nworkers()))
    _b3 = deepcopy(b)
    qrA = DHQR.qr!(A3)
    x3 = qrA \ deepcopy(b)

    @testset "$T distribributed + threaded" begin
      @test norm(A' * A * x3 .- A' * b) < 8stdliberr
    end

    bm3 = @benchmark DHQR.qr!($(_A3)) \ $(_b3)
    tb = minimum(bm3.times) / 1e9

    println("m=$m, n=$n, T=$T: stdlib took $tl seconds")
    println("m=$m, n=$n, T=$T: threaded undistributed took $(ta/tl) times longer")
    println("m=$m, n=$n, T=$T: threaded & distributed took $(tb/tl) times longer")
  end
end


