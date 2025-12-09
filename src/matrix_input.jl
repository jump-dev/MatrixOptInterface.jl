# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

abstract type AbstractLPForm{T} <: MOI.ModelLike end

# FIXME Taken from Polyhedra.jl, we should maybe move this to MOIU
function structural_nonzero_indices(a::SparseArrays.SparseVector)
    return SparseArrays.nonzeroinds(a)
end
structural_nonzero_indices(a::AbstractVector) = eachindex(a)

function _dot_terms(a::AbstractVector{T}) where {T}
    return MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm{T}(a[i], MOI.VariableIndex(i)) for
        i in structural_nonzero_indices(a) if !iszero(a[i])
    ]
end

function _dot(a::AbstractVector{T}) where {T}
    return MOI.ScalarAffineFunction(_dot_terms(a), zero(T))
end

function MOI.get(model::AbstractLPForm, ::MOI.ListOfModelAttributesSet)
    list = MOI.AbstractModelAttribute[]
    push!(list, MOI.ObjectiveSense())
    if model.sense != MOI.FEASIBILITY_SENSE
        push!(
            list,
            MOI.ObjectiveFunction{MOI.get(model, MOI.ObjectiveFunctionType())}(),
        )
    end
    return list
end

MOI.get(model::AbstractLPForm, ::MOI.ObjectiveSense) = model.sense

function MOI.get(
    model::AbstractLPForm{T},
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
) where {T}
    return _dot(model.c)
end

function MOI.get(
    model::AbstractLPForm{T},
    ::MOI.ObjectiveFunctionType,
) where {T}
    return MOI.ScalarAffineFunction{T}
end

function MOI.get(::AbstractLPForm, ::MOI.ListOfVariableAttributesSet)
    return MOI.AbstractVariableAttribute[]
end

MOI.get(model::AbstractLPForm, ::MOI.NumberOfVariables) = length(model.c)

function MOI.get(model::AbstractLPForm, ::MOI.ListOfVariableIndices)
    # TODO return `collect` with MOI v0.9.15 (see
    # https://github.com/jump-dev/MathOptInterface.jl/pull/1110)
    return collect(
        MOIU.LazyMap{MOI.VariableIndex}(
            i -> MOI.VariableIndex(i), # FIXME `LazyMap` needs a `Function` so cannot just give `MOI.VariableIndex`
            1:MOI.get(model, MOI.NumberOfVariables()),
        ),
    )
end

function MOI.get(::AbstractLPForm, ::MOI.ListOfConstraintAttributesSet)
    return MOI.AbstractConstraintAttribute[]
end

function MOI.get(
    model::AbstractLPForm{T},
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}},
) where {T}
    return _dot(model.A[ci.value, :])
end

# We cannot use MOI.Utilities.Hyperrectangle in case the vector type is not `Vector`
struct Hyperrectangle{T,LT<:AbstractVector{T},UT<:AbstractVector{T}} <: MOI.Utilities.AbstractVectorBounds
    lower::LT
    upper::UT
end

MOI.Utilities.function_constants(::Hyperrectangle{T}, row) where {T} = zero(T)

# TODO specialize for SparseVector
function linear_function(c::AbstractVector{T}) where {T}
    return MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(c[i], MOI.VariableIndex(i)) for i in eachindex(c)],
        zero(T),
    )
end

function linear_objective(sense::MOI.OptimizationSense, c::AbstractVector{T}) where {T}
    model = MOI.Utilities.ObjectiveContainer{T}()
    MOI.set(model, MOI.ObjectiveSense(), sense)
    func = linear_function(c)
    MOI.set(model, MOI.ObjectiveFunction{typeof(func)}(), func)
    return model
end

# /!\ Type piracy
function MOI.Utilities.extract_function(
    A::Matrix{T},
    row::Integer,
    constant::T,
) where {T}
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{T}[], constant)
    for col in axes(A, 2)
        val = A[row, col]
        if !iszero(val)
            push!(
                func.terms,
                MOI.ScalarAffineTerm(val, MOI.VariableIndex(col)),
            )
        end
    end
    return func
end

# Copy-paste of MOI.Utilities.MutableSparseMatrixCSC
function _first_in_column(
    A::SparseMatrixCSC,
    row::Integer,
    col::Integer,
)
    range = SparseArrays.nzrange(A, col)
    idx = searchsortedfirst(view(A.rowval, range), row)
    return get(range, idx, last(range) + 1)
end

# Copy-paste of MOI.Utilities.MutableSparseMatrixCSC
function MOI.Utilities.extract_function(
    A::SparseMatrixCSC{T},
    row::Integer,
    constant::T,
) where {T}
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{T}[], constant)
    for col in 1:A.n
        idx = _first_in_column(A, row, col)
        if idx > last(SparseArrays.nzrange(A, col))
            continue
        end
        r = A.rowval[idx]
        if r == row
            push!(
                func.terms,
                MOI.ScalarAffineTerm(A.nzval[idx], MOI.VariableIndex(col)),
            )
        end
    end
    return func
end

function free_variables(n, ::Type{T}) where {T}
    model = MOI.Utilities.VariablesContainer{T}()
    for _ in 1:n
        MOI.add_variable(model)
    end
    return model
end

function nonnegative_variables(n, ::Type{T}) where {T}
    model = MOI.Utilities.VariablesContainer{T}()
    for _ in 1:n
        vi = MOI.add_variable(model)
        MOI.add_constraint(model, vi, MOI.GreaterThan(zero(T)))
    end
    return model
end

function interval_variables(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T}
    @assert eachindex(lower) == eachindex(upper)
    model = MOI.Utilities.VariablesContainer{T}()
    for i in eachindex(lower)
        vi = MOI.add_variable(model)
        MOI.add_constraint(model, vi, MOI.GreaterThan(lower[i]))
        MOI.add_constraint(model, vi, MOI.LessThan(upper[i]))
    end
    return model
end

MOI.Utilities.@product_of_sets(
    EqualTos,
    MOI.EqualTo{T},
)

function equality_constraints(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    sets = EqualTos{T}()
    for _ in eachindex(b)
        MOI.Utilities.add_set(sets, MOI.Utilities.set_index(sets, MOI.EqualTo{T}))
    end
    MOI.Utilities.final_touch(sets)
    constants = Hyperrectangle(b, b)
    model = MOI.Utilities.MatrixOfConstraints{T}(A, constants, sets)
    model.final_touch = true
    return model
end

MOI.Utilities.@product_of_sets(
    LessThans,
    MOI.LessThan{T},
)

function lessthan_constraints(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    sets = LessThans{T}()
    for _ in eachindex(b)
        MOI.Utilities.add_set(sets, MOI.Utilities.set_index(sets, MOI.LessThan{T}))
    end
    MOI.Utilities.final_touch(sets)
    constants = Hyperrectangle(FillArrays.Zeros{T}(length(b)), b)
    model = MOI.Utilities.MatrixOfConstraints{T}(A, constants, sets)
    model.final_touch = true
    return model
end

MOI.Utilities.@product_of_sets(
    Intervals,
    MOI.Interval{T},
)

function interval_constraints(A::AbstractMatrix{T}, lower::AbstractVector{T}, upper::AbstractVector{T}) where {T}
    sets = Intervals{T}()
    for _ in eachindex(lower)
        MOI.Utilities.add_set(sets, MOI.Utilities.set_index(sets, MOI.Interval{T}))
    end
    MOI.Utilities.final_touch(sets)
    constants = Hyperrectangle(lower, upper)
    model = MOI.Utilities.MatrixOfConstraints{T}(A, constants, sets)
    model.final_touch = true
    return model
end

MOI.Utilities.@mix_of_scalar_sets(
    MixedLinearSets,
    MOI.EqualTo{T},
    MOI.GreaterThan{T},
    MOI.LessThan{T},
)

function mix_of_constraints(A::AbstractMatrix{T}, b::AbstractVector{T}, senses::Vector{ConstraintSense}) where {T}
    @assert eachindex(b) == eachindex(senses)
    sets = MixedLinearSets{T}()
    for sense in senses
        # Int(EQUAL_TO) is 0 but `MOI.Utilities.set_index(sets, MOI.EqualTo{T})` is 1
        MOI.Utilities.add_set(sets, Int(sense) + 1)
    end
    MOI.Utilities.final_touch(sets)
    constants = Hyperrectangle(b, b)
    model = MOI.Utilities.MatrixOfConstraints{T}(A, constants, sets)
    model.final_touch = true
    return model
end

"""
    lp_standard_form(sense::MOI.OptimizationSense, c::AbstractVector, A::AbstractMatrix, b::AbstractVector)

Represents a problem of the form:
```
sense ⟨c, x⟩
s.t.  A x == b
      x ≥ 0
```
"""
function lp_standard_form(sense::MOI.OptimizationSense, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert length(c) == n
    @assert length(b) == m
    return MOI.Utilities.GenericModel{T}(
        linear_objective(sense, c),
        nonnegative_variables(n, T),
        equality_constraints(A, b),
    )
end

"""
    lp_geometric_form(sense::MOI.OptimizationSense, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}

Represents a linear problem of the form:
```
sense ⟨c, x⟩
s.t.  Ax <= b
```
"""
function lp_geometric_form(sense::MOI.OptimizationSense, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert length(c) == n
    @assert length(b) == m
    return MOI.Utilities.GenericModel{T}(
        linear_objective(sense, c),
        free_variables(n, T),
        lessthan_constraints(A, b),
    )
end

"""
    function lp_form(
        sense::MOI.OptimizationSense,
        c::AbstractVector{T},
        A::AbstractMatrix{T},
        c_lb::AbstractVector{T},
        c_ub::AbstractVector{T},
        v_lb::AbstractVector{T},
        v_ub::AbstractVector{T},
    ) where {T}

Represents a problem of the form:
```
sense ⟨c, x⟩
s.t.  c_lb <= Ax <= c_ub
      v_lb <=  x <= v_ub
```
"""
function lp_form(
    sense::MOI.OptimizationSense,
    c::AbstractVector{T},
    A::AbstractMatrix{T},
    c_lb::AbstractVector{T},
    c_ub::AbstractVector{T},
    v_lb::AbstractVector{T},
    v_ub::AbstractVector{T},
) where {T}
    m, n = size(A)
    @assert length(c) == n
    @assert length(v_lb) == n
    @assert length(v_ub) == n
    @assert length(c_lb) == m
    @assert length(c_ub) == m
    return MOI.Utilities.GenericModel{T}(
        linear_objective(sense, c),
        interval_variables(v_lb, v_ub),
        interval_constraints(A, c_lb, c_ub),
    )
end

"""
    lp_solver_form(
        sense::MOI.OptimizationSense,
        c::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        sense::Vector{ConstraintSense},
        v_lb::AbstractVector{T},
        v_ub::AbstractVector{T},
    ) where {T}

Represents a problem of the form:
```
sense ⟨c, x⟩
s.t. Ax senses b
     v_lb <=  x <= v_ub
```
"""
function lp_solver_form(
    sense::MOI.OptimizationSense,
    c::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    senses::Vector{ConstraintSense},
    v_lb::AbstractVector{T},
    v_ub::AbstractVector{T},
) where {T}
    m, n = size(A)
    @assert length(c) == n
    @assert length(v_lb) == n
    @assert length(v_ub) == n
    @assert length(b) == m
    return MOI.Utilities.GenericModel{T}(
        linear_objective(sense, c),
        interval_variables(v_lb, v_ub),
        mix_of_constraints(A, b, senses),
    )
end

"""
    MILP{T, LP<:AbstractLPForm{T}}

A mixed-integer problem represented by a linear problem of type `LP`
and a vector indicating each `VariableType`.
"""
struct MILP{T,LP<:MOI.ModelLike}
    lp::LP
    variable_type::Vector{VariableType}
end
