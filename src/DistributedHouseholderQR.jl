module DistributedHouseholderQR

using Distributed, LinearAlgebra, DistributedArrays, SharedArrays, Polyester

LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

alphafactor(x::Real) = -sign(x)
alphafactor(x::Complex) = -exp(im * angle(x))

localcols(m::AbstractMatrix) = 1:size(m, 2)
localcols(m::DArray) = DistributedArrays.localindices(m)[2]

localindexes(A::AbstractArray) = Tuple(1:i for i in size(A))
localindexes(A::DArray) = DistributedArrays.localindices(A)
localindexes(A::SharedArray) = SharedArrays.localindices(A)


rowcolumnblocks(A::AbstractArray, p, i) = (@assert p == 1; return 1:size(A, i))
rowcolumnblocks(A::DArray, p, i) = A.indices[p - A.pids[1] + 1][i]
rowblocks(A, p) = rowcolumnblocks(A, p, 1)
columnblocks(A, p) = rowcolumnblocks(A, p, 2)
Distributed.procs(::AbstractArray) = 1

localblock(A::DArray) = localpart(A)
localblock(A::SharedArray) = A
localblock(A::AbstractArray) = A

isdistributedbycolumns(A) = true#false
function isdistributedbycolumns(A::DArray)
  for p in A.pids
    rowblocks(A, p) == 1:size(A, 1) || return false
  end
  return true
end
isdistributedbyrows(A) = !isdistributedbycolumns(A)

abstract type AbstractLocalBlock end

struct LocalColumnBlock{T} <: AbstractLocalBlock
  Al::T
  Δi::Int
  Δj::Int
  rowrange::UnitRange{Int}
  colrange::UnitRange{Int}
end
function LocalColumnBlock(A::AbstractMatrix)
  rowrange, colrange = localindexes(A)
  @assert rowrange == 1:size(A, 1)
  Δi = rowrange[1] - 1
  Δj = colrange[1] - 1
  return LocalColumnBlock(localblock(A), Δi, Δj, rowrange, colrange)
end
#function LocalColumnBlock(A::AbstractVector)
#  colrange = localindexes(H)[2]
#  Δj = colrange[1] - 1
#  return LocalColumnBlock(localblock(A), Δj, colrange)
#end
Base.setindex!(lcb::LocalColumnBlock, v::Number, i, j) = (lcb.Al[i, j - lcb.Δj] = v)
#Base.setindex!(lcb::LocalColumnBlock, v, j) = (lcb.Al[j - lcb.Δj] = v)
Base.getindex(lcb::LocalColumnBlock, i, j) = lcb.Al[i, j - lcb.Δj]
#Base.getindex(lcb::LocalColumnBlock, j) = lcb.Al[j - lcb.Δj]
Base.eltype(lcb::LocalColumnBlock) = eltype(lcb.Al)
Base.view(lcb::LocalColumnBlock, i, j) = view(lcb.Al, i, j - lcb.Δj)

struct LocalRowBlock{T} <: AbstractLocalBlock
  Al::T
  Δi::Int
  rowrange::UnitRange{Int}
end
function LocalRowBlock(A::AbstractMatrix)
  rowrange, colrange = localindexes(A)
  @assert colrange == 1:size(A, 2) colrange size(A)
  Δi = rowrange[1] - 1
  return LocalRowBlock(localblock(A), Δi, rowrange)
end
function LocalRowBlock(A::AbstractVector)
  rowrange = localindexes(H)[1]
  Δi = rowrange[1] - 1
  return LocalRowBlock(localblock(A), Δi, rowrange)
end
Base.setindex!(lrb::LocalRowBlock, v::Number, i, j) = (lrb.Al[i - lrb.Δi, j] = v)
Base.setindex!(lrb::LocalRowBlock, v, j) = (lrb.Al[i - lrb.Δi] = v)
Base.getindex(lrb::LocalRowBlock, i, j) = lrb.Al[i - lrb.Δi, j]
Base.getindex(lrb::LocalRowBlock, i) = lrb.Al[i - lrb.Δi, j]
Base.eltype(lrb::LocalRowBlock) = eltype(lrb.Al)
Base.view(lrb::LocalRowBlock, i, j) = view(lrb.Al, i - lrb.Δ, j)

function LocalBlock(A)
  rowrange, colrange = localindexes(A)
  rowrange == 1:size(A, 1) && return LocalColumnBlock(A)
  colrange == 1:size(A, 2) && return LocalRowBlock(A)
  throw(ArgumentError("Distribution of indices not implementd"))
end


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

function householder!(A, α)
  for p in procs(A)
    A, α = fetch(@spawnat p _householder!(A, α))
  end
  (A, α)
end

_householder!(H, α) = _householder!(H, α, LocalBlock(H))

function _householder!(H, α, Hl::LocalColumnBlock)
  m, n = size(H)
  Hj = zeros(eltype(H), m)
  t1a = t1b = 0.0
  @inbounds @views for j in Hl.colrange
    t1a += @elapsed begin
    s = norm(Hl[j:m, j])
    α[j] = s * alphafactor(Hl[j, j])
    f = 1 / sqrt(s * (s + abs(Hl[j, j])))
    Hl[j, j] -= α[j]
    for i in j:m
      Hl[i, j] *= f
    end
    end
    t1b += @elapsed begin
    @. Hj = Hl[:, j] # copying this will make all data in end loop local
    @sync for p in procs(H) # this is most expensive
      #j > columnblocks(H, p)[end] && continue
      @spawnat p _householder_inner_localcolumnblock!(H, j, Hj)
    end
    end
  end
  #@show t1a, t1b
  return (H, α)
end

function _householder!(H, α, Hl::LocalRowBlock)
  m, n = size(H)
  Hj = zeros(eltype(H), m)
  t1a = t1b = 0.0
  @inbounds @views for j in intersect(1:n, Hl.rowrange)
    t1a += @elapsed begin
    s = norm(Hl[j:m, j])
    α[j] = s * alphafactor(Hl[j, j])
    f = 1 / sqrt(s * (s + abs(Hl[j, j])))
    Hl[j, j] -= α[j]
    #for i in j:m
    #  Hl[i, j] *= f
    #end
    @sync for p in procs(H) # this is most expensive
      @spawnat p begin
        Hp = LocalRowBlock(H)
        is = intersect(j:m, Hp.rowrange)
        for i in is
          Hl[i, j] *= f
        end
      end
    end
    end
    t1b += @elapsed begin
    @. Hj = H[:, j] # dog slow?
    @sync for p in procs(H) # this is most expensive
      @spawnat p begin
        Hp = LocalRowBlock(H)
        @batch per=core minbatch=64 for jj in j+1:n
          s = dot(view(Hj, j:m), view(H, j:m, jj)) # dog slow?
          @inbounds for i in j:m
            Hp[i, jj] -= Hj[i] * s
          end
        end
      end
    end
    end
  end
  #@show t1a, t1b
  return (H, α)
end

function _householder_inner_localcolumnblock!(H, j, Hj)
  m, n = size(H)
  Hl = LocalColumnBlock(H)
  @batch per=core minbatch=64 for jj in intersect(j+1:n, Hl.colrange)
    s = dot(view(Hj, j:m), view(Hl, j:m, jj))
    @inbounds for i in j:m
      Hl[i, jj] -= Hj[i] * s
    end
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
  return _solve_householder1_inner!(b, H, α, LocalBlock(H))
end

function _solve_householder1_inner!(b, H, α, Hl::LocalColumnBlock)
  m, n = size(H)
  # multuply by Q' ...
  @views for j in intersect(1:n, Hl.colrange)
    s = dot(Hl[j:m, j], b[j:m])
#    s = conj(partialdot(b, Hl, j:m, j))
    @batch per=core minbatch=64 for i in j:m
      @inbounds b[i] -= Hl[i, j] * s
    end
  end
end

function _solve_householder1_inner!(b, H, α, Hl::LocalRowBlock)
  m, n = size(H)
  # multuply by Q' ...
  @views for j in 1:n
    s = dot(H[j:m, j], b[j:m]) # slow
    for p in procs(H)
      wait(@spawnat p begin
        Hl = LocalRowBlock(H)
        for i in intersec(j:m, Hl.rowrange)
          @inbounds b[i] -= Hl[i, j] * s
        end
      end)
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
  if isdistributedbycolumns(H)
    return _solve_householder2_columns!(b::SharedArray, H, α)
  elseif isdistributedbyrows(H)
    return _solve_householder2_rows!(b::SharedArray, H, α)
  else
    throw(error("Not implemented"))
  end
end
function _solve_householder2_columns!(b::SharedArray, H, α)

  m, n = size(H)
  ps = procs(H)
  ps = length(ps) == 1 ? ps : reverse(ps)
  @inbounds @views for i in n:-1:1
    for p in ps
      wait(@spawnat p begin
        Hl = LocalColumnBlock(H)
        js = intersect(i+1:n, Hl.colrange)
        isempty(js) && return b
        bi = zero(promote_type(eltype(b), eltype(Hl)))
        @batch per=core minbatch=64 reduction=(+,bi) for j in js
          @inbounds bi += Hl[i, j] * b[j]
        end
        b[i] -= bi
      end)
    end
    b[i] /= α[i]
  end
  return b
end
function _solve_householder2_rows!(b::SharedArray, H, α)
  m, n = size(H)
  ps = procs(H)
  ps = length(ps) == 1 ? ps : reverse(ps)
  for p in ps
    wait(@spawnat p begin
      Hl = LocalRowBlock(H)
      @inbounds @views for i in intersect(n:-1:1, Hl.rowrange)
        bi = zero(promote_type(eltype(b), eltype(Hl)))
        @batch per=core minbatch=64 reduction=(+,bi) for j in i+1:n
          @inbounds bi += Hl[i, j] * b[j]
        end
        b[i] -= bi
        return b
      end
      b[i] /= α[i]
    end)
  end
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
