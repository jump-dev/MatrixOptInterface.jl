abstract type AbstractLPForm{T} <: MOI.ModelLike end
abstract type AbstractConicForm{T} <: MOI.ModelLike end

# FIXME Taken from Polyhedra.jl, we should maybe move this to MOIU
structural_nonzero_indices(a::SparseArrays.SparseVector) = SparseArrays.nonzeroinds(a)
structural_nonzero_indices(a::AbstractVector) = eachindex(a)
function _dot_terms(a::AbstractVector{T}) where T
    return MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm{T}(a[i], MOI.VariableIndex(i))
        for i in structural_nonzero_indices(a) if !iszero(a[i])
    ]
end
function _dot(a::AbstractVector{T}) where T
    return MOI.ScalarAffineFunction(_dot_terms(a), zero(T))
end

function MOI.get(model::AbstractLPForm, ::MOI.ListOfModelAttributesSet)
    list = MOI.AbstractModelAttribute[]
    push!(list, MOI.ObjectiveSense())
    if model.direction != MOI.FEASIBILITY_SENSE
        push!(list, MOI.ObjectiveFunction{MOI.get(model, MOI.ObjectiveFunctionType())}())
    end
    return list
end
MOI.get(model::AbstractLPForm, ::MOI.ObjectiveSense) = model.direction
function MOI.get(model::AbstractLPForm{T},
                 ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}) where T
    return _dot(model.c)
end
function MOI.get(model::AbstractLPForm{T}, ::MOI.ObjectiveFunctionType) where T
    return MOI.ScalarAffineFunction{T}
end

MOI.get(::AbstractLPForm, ::MOI.ListOfVariableAttributesSet) = MOI.AbstractVariableAttribute[]
MOI.get(model::AbstractLPForm, ::MOI.NumberOfVariables) = length(model.c)
function MOI.get(model::AbstractLPForm, ::MOI.ListOfVariableIndices)
    # TODO return `collect` with MOI v0.9.15 (see https://github.com/jump-dev/MathOptInterface.jl/pull/1110)
    return collect(MOIU.LazyMap{MOI.VariableIndex}(
        i -> MOI.VariableIndex(i), # FIXME `LazyMap` needs a `Function` so cannot just give `MOI.VariableIndex`
        1:MOI.get(model, MOI.NumberOfVariables())
    ))
end

MOI.get(::AbstractLPForm, ::MOI.ListOfConstraintAttributesSet) = MOI.AbstractConstraintAttribute[]
function MOI.get(model::AbstractLPForm{T}, ::MOI.ConstraintFunction,
                 ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}}) where T
    return _dot(model.A[ci.value, :])
end


struct LPStandardForm{T, AT<:AbstractMatrix{T}} <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::AT
    b::Vector{T}
end

function MOI.get(model::LPStandardForm{T}, ::MOI.ListOfConstraints) where T
    return [(MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
            (MOI.VectorOfVariables, MOI.Nonnegatives)]
end
const EQ{T} = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}
function MOI.get(
    model::LPStandardForm{T},
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}
) where T
    # TODO return `collect` with MOI v0.9.15 (see https://github.com/jump-dev/MathOptInterface.jl/pull/1110)
    return collect(MOIU.LazyMap{EQ{T}}(
        i -> EQ{T}(i), # FIXME `LazyMap` needs a `Function` so cannot just give `EQ{T}`
        1:size(model.A, 1)
    ))
end
function MOI.get(model::LPStandardForm, ::MOI.ConstraintSet,
                 ci::EQ)
    return MOI.EqualTo(model.b[ci.value])
end
const NONNEG = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}
function MOI.get(
    model::LPStandardForm,
    ::MOI.ListOfConstraintIndices{
        MOI.VectorOfVariables, MOI.Nonnegatives}
)
    return [NONNEG(1)]
end
function MOI.get(model::LPStandardForm, ::MOI.ConstraintFunction,
                 ci::NONNEG)
    return MOI.VectorOfVariables(MOI.get(model, MOI.ListOfVariableIndices()))
end
function MOI.get(model::LPStandardForm, ::MOI.ConstraintSet,
                 ci::NONNEG)
    return MOI.Nonnegatives(MOI.get(model, MOI.NumberOfVariables()))
end

struct LPGeometricForm{T, AT<:AbstractMatrix{T}} <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::AT
    b::Vector{T}
end

function MOI.get(model::LPGeometricForm{T}, ::MOI.ListOfConstraints) where T
    return [(MOI.ScalarAffineFunction{T}, MOI.LessThan{T})]
end
const LT{T} = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}
function MOI.get(
    model::LPGeometricForm{T},
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}
) where T
    # FIXME `copy_constraint` needs a `Vector`
    return collect(MOIU.LazyMap{LT{T}}(
        i -> LT{T}(i), # FIXME `LazyMap` needs a `Function` so cannot just give `LT{T}`
        1:size(model.A, 1))
    )
end
function MOI.get(model::LPGeometricForm, ::MOI.ConstraintSet,
                 ci::LT)
    return MOI.LessThan(model.b[ci.value])
end

abstract type LPMixedForm{T} <: AbstractLPForm{T} end

function MOI.get(model::LPMixedForm{T}, ::MOI.ListOfConstraints) where T
    list = Tuple{DataType, DataType}[]
    for S in [MOI.EqualTo{T}, MOI.Interval{T}, MOI.GreaterThan{T}, MOI.LessThan{T}]
        for F in [MOI.SingleVariable, MOI.ScalarAffineFunction{T}]
            if !iszero(MOI.get(model, MOI.NumberOfConstraints{F, S}()))
                push!(list, (F, S))
            end
        end
    end
    return list
end

struct BoundSense <: MOI.AbstractVariableAttribute end
function MOI.get(model::LPMixedForm, ::BoundSense, vi::MOI.VariableIndex)
    return _bound_sense(model.v_lb[vi.value], model.v_ub[vi.value])
end

const LinearBounds{T} = Union{MOI.EqualTo{T}, MOI.Interval{T}, MOI.GreaterThan{T}, MOI.LessThan{T}}

function MOI.get(model::LPMixedForm{T}, ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{T}, S}) where {T, S <: LinearBounds{T}}
    s = _sense(S)
    return count(1:size(model.A, 1)) do i
        _constraint_bound_sense(model, i) == s
    end
end
const AFF{T, S} = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, S}
function MOI.get(model::LPMixedForm{T}, ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, S}) where {T, S <: LinearBounds{T}}
    s = _sense(S)
    # FIXME `copy_constraint` needs a `Vector`
    return collect(MOIU.LazyMap{AFF{T, S}}(
        i -> AFF{T, S}(i), # FIXME `LazyMap` needs a `Function` so cannot just give `AFF{T, S}`
        Base.Iterators.Filter(1:size(model.A, 1)) do i
            _constraint_bound_sense(model, i) == s
        end
    ))
end

const VBOUND{S} = MOI.ConstraintIndex{MOI.SingleVariable, S}
function MOI.get(model::LPMixedForm{T}, ::MOI.NumberOfConstraints{MOI.SingleVariable, S}) where {T, S <: LinearBounds{T}}
    s = _sense(S)
    return count(MOI.get(model, MOI.ListOfVariableIndices())) do vi
        MOI.get(model, BoundSense(), vi) == s
    end
end
function MOI.get(model::LPMixedForm{T}, ::MOI.ListOfConstraintIndices{MOI.SingleVariable, S}) where {T, S <: LinearBounds{T}}
    s = _sense(S)
    return collect(MOIU.LazyMap{VBOUND{S}}(
        Base.Iterators.Filter(MOI.get(model, MOI.ListOfVariableIndices())) do vi
            MOI.get(model, BoundSense(), vi) == s
        end
    ) do vi
        VBOUND{S}(vi.value)
    end)
end
function MOI.get(model::LPMixedForm, ::MOI.ConstraintFunction, ci::VBOUND)
    return MOI.SingleVariable(MOI.VariableIndex(ci.value))
end
function MOI.get(model::LPMixedForm, ::MOI.ConstraintSet, ci::VBOUND)
    return _bound_set(model.v_lb[ci.value], model.v_ub[ci.value])
end

struct LPForm{T, AT<:AbstractMatrix{T}} <: LPMixedForm{T}#, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::AT
    c_lb::Vector{T}
    c_ub::Vector{T}
    v_lb::Vector{T}
    v_ub::Vector{T}
end

function _constraint_bound_sense(model::LPForm, i)
    return _bound_sense(model.c_lb[i], model.c_ub[i])
end
function MOI.get(model::LPForm, ::MOI.ConstraintSet, ci::AFF)
    return _bound_set(model.c_lb[ci.value], model.c_ub[ci.value])
end

struct LPSolverForm{T, AT<:AbstractMatrix{T}} <: LPMixedForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::AT
    b::Vector{T}
    senses::Vector{ConstraintSense}
    v_lb::Vector{T}
    v_ub::Vector{T}
end

function _constraint_bound_sense(model::LPSolverForm, i)
    return model.senses[i]
end
function MOI.get(model::LPSolverForm, ::MOI.ConstraintSet, ci::AFF)
    s = model.senses[ci.value]
    β = model.b[ci.value]
    if s == GREATER_THAN
        return MOI.GreaterThan(β)
    elseif s == LESS_THAN
        return MOI.LessThan(β)
    else
        s == EQUAL_TO || error("Invalid $s")
        return MOI.EqualTo(β)
    end
end

struct MILP{T, LP<:AbstractLPForm{T}}
    lp::LP
    variable_type::Vector{VariableType}
end

struct ConicForm{T, AT<:AbstractMatrix{T}} <: AbstractConicForm{T}
    # assuming minimization for now
    # direction::MOI.OptimizationSense 
    c::Vector{T}
    A::AT
    b::Vector{T}
    cones::Vector{<: MOI.AbstractVectorSet}
end

"""
Convert a `MathOptInterface` model to `MatrixOptInterface`
"""
function getConicForm(model::M, con_idx) where {M <: MOI.AbstractOptimizer}
    conic = ConicData()

    # 1st allocate variables and constraints
    N = MOI.get(model, MOI.NumberOfVariables())
    __allocate_variables(conic, N)
    for con in con_idx
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        F = typeof(func)
        S = typeof(set)
        __allocate_constraint(conic, func, set)
    end

    __load_variables(conic, N)
    
    CONES_OFFSET = Dict(
        MOI.Zeros => 0,
        MOI.Nonnegatives => 0,
        MOI.SecondOrderCone => 0,
        MOI.PositiveSemidefiniteConeTriangle => 0,
        MOI.ExponentialCone => 0, 
        MOI.DualExponentialCone => 0
    )

    for con in con_idx
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        F = typeof(func)
        S = typeof(set)
        __load_constraint(conic, CI{F, S}(CONES_OFFSET[S]), func, set)
        CONES_OFFSET[S] += cons_offset(set)
    end
    
    # now SCS data shud be allocated
    A = sparse(
        conic.data.I, 
        conic.data.J, 
        conic.data.V 
    )
    b = conic.data.b 

    # extract `c`
    obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    __load(conic, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    c = conic.data.c

    # fix optimization sense
    if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        c = -c
    end

    # reorder constraints
    cis = sort(
        con_idx, 
        by = x->CONES[typeof(MOI.get(model, MOI.ConstraintSet(), x))]
    )

    # extract cones
    cones = MOI.get(model, MOI.ConstraintSet(), cis)

    return ConicForm{Float64, typeof(A)}(
        c, A, b, cones
    )
end
