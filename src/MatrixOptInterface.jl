module MatrixOptInterface

using LinearAlgebra, SparseArrays
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

# export MatrixOptimizer

@enum ConstraintSense EQUAL_TO GREATER_THAN LESS_THAN INTERVAL
_sense(::Type{<:MOI.EqualTo}) = EQUAL_TO
_sense(::Type{<:MOI.LessThan}) = LESS_THAN
_sense(::Type{<:MOI.GreaterThan}) = GREATER_THAN
_sense(::Type{<:MOI.Interval}) = INTERVAL

_no_upper(bound) = bound != typemax(bound)
_no_lower(bound) = bound != typemin(bound)

function _bound_sense(lb::T, ub::T) where {T}
    if _no_upper(ub)
        if _no_lower(lb)
            if lb == ub
                return EQUAL_TO
            else
                return INTERVAL
            end
        else
            return LESS_THAN
        end
    else
        if _no_lower(lb)
            return GREATER_THAN
        else
            return
        end
    end
end

function _bound_set(lb::T, ub::T) where {T}
    if _no_upper(ub)
        if _no_lower(lb)
            if ub == lb
                return MOI.EqualTo(lb)
            else
                return MOI.Interval(lb, ub)
            end
        else
            return MOI.LessThan(ub)
        end
    else
        if _no_lower(lb)
            return MOI.GreaterThan(lb)
        else
            return
        end
    end
end

@enum VariableType CONTINUOUS INTEGER BINARY

include("sparse_matrix.jl")
include("conic_form.jl")
include("matrix_input.jl")
include("change_form.jl")

end
