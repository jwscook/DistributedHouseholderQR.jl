module DistributedHouseholderQR

using Distributed, LinearAlgebra, DistributedArrays, SharedArrays, Polyester
using SIMD, Base.Threads

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

alphafactor(x::Real) = -sign(x)
alphafactor(x::Complex) = -exp(im * angle(x))

localcols(m::AbstractMatrix) = 1:size(m, 2)
localcols(m::DArray) = DistributedArrays.localindices(m)[2]

localindexes(A::AbstractArray) = Tuple(1:i for i in size(A))
localindexes(A::DArray) = DistributedArrays.localindices(A)
localindexes(A::SharedArray) = SharedArrays.localindices(A)

columnblocks(m::AbstractArray, n) = (@assert n == 1; return 1:size(m, 2))
columnblocks(m::DArray, n) = m.indices[n - m.pids[1] + 1][2]
Distributed.procs(::AbstractArray) = 1

localblock(A::DArray) = localpart(A)
localblock(A::SharedArray) = A
localblock(A::AbstractArray) = A

struct LocalColumnBlock{T}
  Al::T
  Δj::Int
  colrange::UnitRange{Int}
end
function LocalColumnBlock(A::AbstractMatrix)
  rowrange, colrange = localindexes(A)
  @assert rowrange == 1:size(A, 1)
  Δj = colrange[1] - 1
  return LocalColumnBlock(localblock(A), Δj, colrange)
end
Base.setindex!(lcb::LocalColumnBlock, v::Number, i, j) = (lcb.Al[i, j - lcb.Δj] = v)
Base.getindex(lcb::LocalColumnBlock, i, j) = lcb.Al[i, j - lcb.Δj]
Base.eltype(lcb::LocalColumnBlock) = eltype(lcb.Al)
Base.view(lcb::LocalColumnBlock, i, j) = view(lcb.Al, i, j - lcb.Δj)

@inline function partialdot(a, b, is, ::Type{<:Real})
 # this will be called in a thread
  s = zero(promote_type(eltype(b), eltype(b)))
  @inbounds @simd for i in is
    s += a[i] * b[i]
  end
  return s
end

@inline function partialdot(a, b, is, ::Type{<:Complex})
  s = zero(promote_type(eltype(b), eltype(b)))
  @inbounds for i in is
    ar, ai = reim(a[i])
    br, bi = reim(b[i])
    s += Complex(ar * br + ai * bi, ar * bi - ai * br)
  end
  return s
end


using StrideArraysCore
Base.@propagate_inbounds SIMD._pointer(arr::StrideArraysCore.PtrArray, i, I) =
    pointer(Base.unsafe_view(arr, 1, I...), i)

#  # This is dog slow unfortunately
#@inline function partialdotsimdvectors(::Val{4}, ::Type{T}) where T
#  shuffle1 = Val{(0, 0, 2, 2)}() # 4 long
#  shuffle2 = Val{(1, 1, 3, 3)}()
#  shuffle3 = Val{(1, 0, 3, 2)}()
#  z = one(T)
#  v = Vec((z, -z, z, -z))
#  return (shuffle1, shuffle2, shuffle3, v)
#end
#@inline partialdot(a, b, is, ::Type{T}) where T  = partialdot(a, b, is)
#@inline function partialdot(a, b, is, ::Type{ComplexF64})
#  return partialdot(a, b, is, ComplexF64, Val(4))
#end
#@inline function sumcollapse(v::Vec{N, T}) where {N, T}
#  rv = iv = zero(T)
#  for i in 1:2:N
#    rv += v[i]
#    iv += v[i + 1]
#  end
#  return Complex(rv, iv)
#end
#@inline function partialdot(a, b, is, ::Type{ComplexF64}, ::Val{N}) where N
#  ria = reinterpret(real(eltype(a)), view(a, is))
#  rib = reinterpret(real(eltype(b)), view(b, is))
#
#  slimit = (length(is) ÷ N) * N # 4 Float64s per register
#  lane = VecRange{N}(0)
#  T = real(promote_type(eltype(a), eltype(b)))
#  svec = Vec(Tuple(zero(T) for _ in 1:N))
#  sh1, sh2, sh3, v = partialdotsimdvectors(Val(N), T)
#  @inbounds for ii in 1:N÷2:slimit - 1
#    i = (N÷2)*ii # i = 2, 4, 8, ...: i-1 for real, i for imag
#    # conj(a) * b = (ar - im * ai) * (br + im * bi) =    (ar*br + ai*bi)
#    #                                                 im*(ar*bi - ai*br)
#    rialane = ria[i-1 + lane]
#    riblane = rib[i-1 + lane]
#    svec = muladd(shufflevector(rialane, sh1), riblane, svec)
#    stmp = shufflevector(rialane, sh2) * shufflevector(riblane, sh3)
#    svec = muladd(v, stmp, svec)
#  end
#  s = sumcollapse(svec)
#  @inbounds for i in minimum(is) + slimit:maximum(is)
#    s += conj(a[i]) * b[i]
#  end
#  return s
#end

householder!(A, α) = _householder!(A, α)

function householder!(A::DArray, α)
  for p in procs(A)
    A, α = fetch(@spawnat p _householder!(A, α))
  end
  (A, α)
end

function _householder!(H, α)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  Hj = Vector(zeros(eltype(H), size(H, 1)))
  t1a = t1b = 0.0
  @inbounds for j in Hl.colrange
    t1a += @elapsed begin
    s = norm(view(Hl, j:m, j))
    α[j] = s * alphafactor(Hl[j, j])
    f = 1 / sqrt(s * (s + abs(Hl[j, j])))
    Hl[j, j] -= α[j]
    @batch per=core minbatch=64 for i in j:m
      Hl[i, j] *= f
    end
    end
    t1b += @elapsed begin
    @batch per=core minbatch=64 for i in eachindex(Hj)
      Hj[i] = Hl[i, j] # copying this will make all data in end loop local
    end
    @sync for p in procs(H) # this is most expensive
      @spawnat p _householder_inner!(H, j, Hj)
    end
    end
  end
  #@show t1a, t1b
  return (H, α)
end

@inline function hotloop!(Hl, Hj, s, is, jj)
  @inbounds @views for i in is
    Hl[i, jj] -= Hj[i] * s
  end
end

@inline function hotloop!(Hl, Hj, s, is, jj, ::Type{<:Real})
  @inbounds @simd for i in is
    Hl[i, jj] -= Hj[i] * s
  end
end

function hotloopsimdvectors(s, ::Val{4})
  sr, si = reim(s)
  shuffle1 = Val{(0, 0, 2, 2)}() # 4 long
  shuffle2 = Val{(1, 1, 3, 3)}()
  s1 = Vec((-sr, -si, -sr, -si))
  s2 = Vec((si, -sr, si, -sr))
  return (shuffle1, shuffle2, s1, s2)
end

@inline function hotloop!(Hl, Hj, s, is, jj, ::Type{<:Complex})
  return hotloop!(Hl, Hj, s, is, jj)
end
@inline function hotloop!(Hl, Hj, s, is, jj, ::Type{ComplexF64})
  return hotloop!(Hl, Hj, s, is, jj, ComplexF64, Val(4))
end
@inline function hotloop!(Hl, Hj, s, is, jj, ::Type{T}, ::Val{N}
    ) where {T<:Complex, N}
  riHl = reinterpret(real(eltype(Hl)), view(Hl, :, jj))
  riHj = reinterpret(real(eltype(Hj)), Hj)

  slimit = (length(is) ÷ N) * N # 4 Float64s per register for 256 bit ymm
  lane = VecRange{N}(0)
  (shuffle1, shuffle2, s1, s2) = hotloopsimdvectors(s, Val(N))
  @inbounds for ii in minimum(is):N÷2:minimum(is) + slimit - 1
    i = N÷2*ii # i = 2, 4, 8, ...: i-1 for real, i for imag
    riHjlane = riHj[i-1 + lane]
    riHllane = riHl[i-1 + lane]
    riHllane = muladd(s1, shufflevector(riHjlane, shuffle1), riHllane)
    riHllane = muladd(s2, shufflevector(riHjlane, shuffle2), riHllane)
    riHl[i-1 + lane] = riHllane
  end
  @inbounds for ii in minimum(is) + slimit:maximum(is)
    Hl[ii, jj] -= s * Hj[ii]
  end
end

function _householder_inner!(H, j, Hj::Vector)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  # @batch makes Hj a StridedVector rather than a Vector, and then SIMD can't cope:
  #@batch per=core minbatch=4 for jj in intersect(j+1:n, Hl.colrange)
  # FLoops can be used the simd version of the hotloop:
  #@floop ThreadedEx() for jj in intersect(j+1:n, Hl.colrange)
  jjs = intersect(j+1:n, Hl.colrange)
  isempty(jjs) && return Hl
  nchunk = ceil(Int, length(jjs) / nthreads())
  @assert nchunk * nthreads() >= length(jjs)
  jjsblocks = [jjs[(i-1)*nchunk+1:min(i*nchunk, length(jjs))] for i in 1:nthreads()]
  @batch per=core minbatch=1 for jjt in jjsblocks
    for jj in jjt
      s = partialdot(Hj, view(Hl, :, jj), j:m, eltype(Hj))
      hotloop!(Hl, Hj, s, j:m, jj, eltype(H))
    end
  end
  return Hl
end

function _solve_householder1!(b::Vector, H, α::Vector)
  m, n = size(H)
  for j in 1:n
    s = partialdot(view(H, :, j), b, j:m, eltype(H))
    @batch for i in j:m
      @inbounds b[i] -= H[i, j] * s
    end
  end
  return b
end

function _solve_householder1!(b::SharedArray, H, α)
  for p in procs(H)
    wait(@spawnat p _solve_householder1_inner!(b, H, α))
  end
end

function _solve_householder1_inner!(b, H, α)
  m, n = size(H)
  # multuply by Q' ...
  Hl = LocalColumnBlock(H)
  @views for j in intersect(1:n, Hl.colrange)
    s = partialdot(view(Hl, :, j), b, j:m, eltype(H))
    @batch per=core minbatch=64 for i in j:m
      @inbounds b[i] -= Hl[i, j] * s
    end
  end
end

function _solve_householder2!(b::Vector, H, α::Vector)
  m, n = size(H)
  @inbounds @views for i in n:-1:1
    bi = b[i]
    @batch per=core minbatch=64 reduction=(-,bi) for j in i+1:n
      bi -= H[i, j] * b[j]
    end
    b[i] = bi / α[i]
  end
  return b
end

function _solve_householder2!(b::SharedArray, H, α)
  m, n = size(H)
  ps = procs(H)
  ps = length(ps) == 1 ? ps : reverse(ps)
  @inbounds @views for i in n:-1:1
    futures = Vector{Future}()
    for p in ps
      i > columnblocks(H, p)[end] && continue
      push!(futures, @spawnat p _solve_householder2_inner!(b, H, i))
    end
    bi = sum(fetch.(futures))
    b[i] = (b[i] - bi) / α[i]
  end
  return b
end

function _solve_householder2_inner!(b, H, i)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  js = intersect(i+1:n, Hl.colrange)
  bi = zero(promote_type(eltype(b), eltype(Hl)))
  isempty(js) && return bi
  for j in js
    @inbounds bi += Hl[i, j] * b[j]
  end
  return bi
end

function solve_householder!(b, H, α)
  # H is the matrix A that has undergone householder!(A, α)
  m, n = size(H)
  # multuply by Q' ...
  _solve_householder1!(b, H, α)
  # now that b holds the value of Q'b
  # we may back sub with R
  t2 = @elapsed _solve_householder2!(b, H, α)
  #@show t2
  return b[1:n]
end

struct DistributedHouseholderQRStruct{T1, T2}
  A::T1
  α::T2
end

function DistributedHouseholderQRStruct(A::DArray)
  α = SharedArray(zeros(eltype(A), size(A, 2)))
  return DistributedHouseholderQRStruct(A, α)
end

function DistributedHouseholderQRStruct(A)
  α = zeros(eltype(A), size(A, 2))
  return DistributedHouseholderQRStruct(A, α)
end

function qr!(A)
  H = DistributedHouseholderQRStruct(A)
  householder!(H.A, H.α)
  return H
end

function LinearAlgebra.:(\)(H::DistributedHouseholderQRStruct, b)
  s = SharedArray(b)
  solve_householder!(s, H.A, H.α)
  return s[1:size(H.A, 2)]
end


end # module DistributedHouseholderQR
