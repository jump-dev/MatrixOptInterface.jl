abstract type AbstractIndexing end
struct ZeroBasedIndexing <: AbstractIndexing end
struct OneBasedIndexing <: AbstractIndexing end

first_index(::ZeroBasedIndexing) = 0
first_index(::OneBasedIndexing) = 1
shift(x, ::ZeroBasedIndexing, ::ZeroBasedIndexing) = x
shift(x::Integer, ::ZeroBasedIndexing, ::OneBasedIndexing) = x + 1
shift(x::Array{<:Integer}, ::ZeroBasedIndexing, ::OneBasedIndexing) = x .+ 1
shift(x::Integer, ::OneBasedIndexing, ::ZeroBasedIndexing) = x - 1
shift(x, ::OneBasedIndexing, ::OneBasedIndexing) = x

mutable struct SparseMatrixCSRtoCSC{Tv,Ti<:Integer,I<:AbstractIndexing}
    indexing::I
    m::Int # Number of rows
    n::Int # Number of columns
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
    function SparseMatrixCSRtoCSC{Tv,Ti,I}(n) where {Tv,Ti<:Integer,I}
        A = new{Tv,Ti,I}()
        A.n = n
        A.colptr = zeros(Ti, n + 1)
        return A
    end
end
function allocate_nonzeros(A::SparseMatrixCSRtoCSC{Tv,Ti}) where {Tv,Ti}
    for i in 3:length(A.colptr)
        A.colptr[i] += A.colptr[i-1]
    end
    A.rowval = Vector{Ti}(undef, A.colptr[end])
    return A.nzval = Vector{Tv}(undef, A.colptr[end])
end
function final_touch(A::SparseMatrixCSRtoCSC)
    for i in length(A.colptr):-1:2
        A.colptr[i] = shift(A.colptr[i-1], ZeroBasedIndexing(), A.indexing)
    end
    return A.colptr[1] = first_index(A.indexing)
end
function _allocate_terms(colptr, indexmap, terms)
    for term in terms
        colptr[indexmap[term.scalar_term.variable].value+1] += 1
    end
end
function allocate_terms(A::SparseMatrixCSRtoCSC, indexmap, func)
    return _allocate_terms(A.colptr, indexmap, func.terms)
end
function _load_terms(colptr, rowval, nzval, indexmap, terms, offset)
    for term in terms
        ptr = colptr[indexmap[term.scalar_term.variable].value] += 1
        rowval[ptr] = offset + term.output_index
        nzval[ptr] = -term.scalar_term.coefficient
    end
end
function load_terms(A::SparseMatrixCSRtoCSC, indexmap, func, offset)
    return _load_terms(
        A.colptr,
        A.rowval,
        A.nzval,
        indexmap,
        func.terms,
        shift(offset, OneBasedIndexing(), A.indexing),
    )
end

"""
    Base.convert(::Type{SparseMatrixCSC{Tv, Ti}}, A::SparseMatrixCSRtoCSC{Tv, Ti, I}) where {Tv, Ti, I}

Converts `A` to a `SparseMatrixCSC`. Note that the field `A.nzval` is **not
copied** so if `A` is modified after the call of this function, it can still
affect the value returned. Moreover, if `I` is `OneBasedIndexing`, `colptr`
and `rowval` are not copied either, i.e., the conversion is allocation-free.
"""
function Base.convert(
    ::Type{SparseMatrixCSC{Tv,Ti}},
    A::SparseMatrixCSRtoCSC{Tv,Ti},
) where {Tv,Ti}
    return SparseMatrixCSC{Tv,Ti}(
        A.m,
        A.n,
        shift(A.colptr, A.indexing, OneBasedIndexing()),
        shift(A.rowval, A.indexing, OneBasedIndexing()),
        A.nzval,
    )
end
