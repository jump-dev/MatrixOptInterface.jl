mutable struct SparseMatrixCSRtoCSC{Tv,Ti<:Integer}
    m::Int # Number of rows
    n::Int # Number of columns
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
    function SparseMatrixCSRtoCSC{Tv, Ti}(n) where {Tv, Ti<:Integer}
        A = new()
        A.n = n
        A.colptr = zeros(Ti, n + 1)
        return A
    end
end
function allocate_nonzeros(A::SparseMatrixCSRtoCSC{Tv, Ti}) where {Tv, Ti}
    for i in 3:length(A.colptr)
        A.colptr[i] += A.colptr[i - 1]
    end
    A.rowval = Vector{Ti}(undef, A.colptr[end])
    A.nzval = Vector{Tv}(undef, A.colptr[end])
end
function final_touch(A::SparseMatrixCSRtoCSC)
    for i in length(A.colptr):-1:2
        A.colptr[i] = A.colptr[i - 1]
    end
    A.colptr[1] = 0
end
function _allocate_terms(colptr, indexmap, terms)
    for term in terms
        colptr[indexmap[term.scalar_term.variable_index].value + 1] += 1
    end
end
function allocate_terms(A::SparseMatrixCSRtoCSC, indexmap, func)
    _allocate_terms(A.colptr, indexmap, func.terms)
end
function _load_terms(colptr, rowval, nzval, indexmap, terms, offset)
    for term in terms
        ptr = colptr[indexmap[term.scalar_term.variable_index].value] += 1
        rowval[ptr] = offset + term.output_index - 1
        nzval[ptr] = -term.scalar_term.coefficient
    end
end
function load_terms(A::SparseMatrixCSRtoCSC, indexmap, func, offset)
    _load_terms(A.colptr, A.rowval, A.nzval, indexmap, func.terms, offset)
end
