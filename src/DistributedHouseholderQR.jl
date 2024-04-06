module DistributedHouseholderQR

using Distributed, LinearAlgebra, DistributedArrays, SharedArrays, Polyester, SIMD

LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

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

@inline function partialdot(v, A, is, j)
  s = if length(is) < 64
    dot(v[is], A[is, j])
  else
    s = zero(eltype(A))
    @inbounds @batch minbatch=64 reduction=(+,s) for i in is
      s += conj(v[i]) * A[i, j]
    end
    s
  end
  return s
end


householder!(A, α) = _householder!(A, α)

function householder!(A::DArray, α)
  for p in procs(A)
    A, α = fetch(@spawnat p _householder!(A, α))
  end
  (A, α)
end

columnbuffer(H) = Vector(zeros(eltype(H), size(H, 1)))

function _householder!(H, α)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  Hj = columnbuffer(H)
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
      #j > columnblocks(H, p)[end] && continue
      @spawnat p _householder_inner!(H, j, Hj)
    end
    end
  end
  @show t1a, t1b
  return (H, α)
end

@inline function hotloop!(Hl, Hj::Vector, s, is, jj)
  @inbounds @views for i in is
    Hl[i, jj] -= Hj[i] * s
  end
end
@inline function hotloop!(Hl, Hj::Vector, s, is, jj, ::Type{<:Real})
  # hand rolled simd loop as a practice for the complex version
  slimit = (length(is) ÷ 4) * 4
  vlimit = Vec(ntuple(i -> maximum(is) - i + 1, 4))
  Hljj = view(Hl, :, jj)
  lane = VecRange{4}(0)
  @inbounds for i in minimum(is):4:maximum(is)
    if i <= slimit
      Hljj[i + lane] -= Hj[i + lane] * s
    else
      mask = Vec{4, Int}(i) <= vlimit
      Hljj[i + lane, mask] -= Hj[i + lane, mask] * s
    end
  end
end

@inline function hotloop!(Hl, Hj::Vector{T}, s, is, jj,
                          ::Type{T}) where {T<:Complex}
  sr, si = reim(s)
  riHl = reinterpret(real(eltype(Hl.Al)), view(Hl.Al, :, jj - Hl.Δj))
  riHj = reinterpret(real(eltype(Hj)), Hj)

  slimit = (length(is) ÷ 4) * 4 # 4 Float64s per register
  vlimit = Vec(ntuple(i -> 2 * (maximum(is) + 1) - i, 4))
  lane = VecRange{4}(0)
  shuffle1 = Val{(0, 0, 2, 2)}() # 4 long
  shuffle2 = Val{(1, 1, 3, 3)}()
  s1 = Vec((sr, si, sr, si))
  s2 = Vec((-si, sr, -si, sr))
  @inbounds for ii in minimum(is):2:maximum(is)
    i = 2ii # i = 2, 4, 8, ...: i-1 for real, i for imag
    if i <= slimit
      riHjlane = riHj[i-1 + lane]
      riHl[i-1 + lane] -= s1 * shufflevector(riHjlane, shuffle1) +
                          s2 * shufflevector(riHjlane, shuffle2)

    else
      mask = Vec{4, Int}(i) <= vlimit
      riHjlane = riHj[i-1 + lane, mask]
      riHl[i-1 + lane, mask] -= s1 * shufflevector(riHjlane, shuffle1) +
                                s2 * shufflevector(riHjlane, shuffle2)
    end
  end
end
using FLoops
function _householder_inner!(H, j, Hj::Vector)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  # @batch makes Hj a StridedVector rather than a Vector, and then SIMD can't cope
  #@batch per=core minbatch=64 for jj in intersect(j+1:n, Hl.colrange)
  for jj in intersect(j+1:n, Hl.colrange)
    s = dot(view(Hj, j:m), view(Hl, j:m, jj))
    hotloop!(Hl, Hj, s, j:m, jj, eltype(H))
    #@inbounds @views for i in j:m
    #  Hl[i, jj] -= Hj[i] * s
    #end
  end
  return Hl
end

function _solve_householder1!(b::Vector, H, α::Vector)
  m, n = size(H)
  for j in 1:n
    s = dot(H[j:m, j], b[j:m])
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
    s = dot(Hl[j:m, j], b[j:m])
#    s = conj(partialdot(b, Hl, j:m, j))
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
    for p in ps
      i > columnblocks(H, p)[end] && continue
      wait(@spawnat p _solve_householder2_inner!(b, H, i))
    end
    b[i] /= α[i]
  end
  return b
end

function _solve_householder2_inner!(b, H, i)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  js = intersect(i+1:n, Hl.colrange)
  isempty(js) && return b
  bi = zero(promote_type(eltype(b), eltype(Hl)))
  @batch per=core minbatch=64 reduction=(+,bi) for j in js
    @inbounds bi += Hl[i, j] * b[j]
  end
  b[i] -= bi
  return b
end

function solve_householder!(b, H, α)
  # H is the matrix A that has undergone householder!(A, α)
  m, n = size(H)
  # multuply by Q' ...
  _solve_householder1!(b, H, α)
  # now that b holds the value of Q'b
  # we may back sub with R
  t2 = @elapsed _solve_householder2!(b, H, α)
  @show t2
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
