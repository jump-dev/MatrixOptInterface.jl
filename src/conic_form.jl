
"""
    GeometricConicForm{T, AT, VT, C} <: MOI.ModelLike

Represents an optimization model of the form:
```
sense ⟨c, x⟩ + c0
s.t.  b_i - A_i x ∈ C_i ∀ i
```
with each `C_i` a cone defined in MOI.
"""
mutable struct GeometricConicForm{T, AT, VB, VC, C} <: MOI.ModelLike
    num_rows::Vector{Int}
    dimension::Dict{Int, Int}
    sense::MOI.OptimizationSense
    objective_constant::T # The objective
    A::Union{Nothing, AT} # The constraints
    b::VB          # `b - Ax in cones`
    c::VC          # `sense c'x + objective_constant`
    cone_types::C
    cone_types_dict::Dict{DataType, Int}

    function GeometricConicForm{T, AT, VB, VC}(cone_types) where {T, AT, VB, VC}
        model = new{T, AT, VB, VC, typeof(cone_types)}()
        model.cone_types = cone_types
        model.cone_types_dict = Dict{DataType, Int}(
            s => i for (i, s) in enumerate(cone_types)
        )
        model.num_rows = zeros(Int, length(cone_types))
        model.dimension = Dict{Int, Int}()
        model.A = nothing
        return model
    end
end

function GeometricConicForm{T, AT, VT}(cone_types) where {T, AT, VT}
    return GeometricConicForm{T, AT, VT, VT}(cone_types)
end

_set_type(::MOI.ConstraintIndex{F,S}) where {F,S} = S

MOI.is_empty(model::GeometricConicForm) = model.A === nothing

function MOI.empty!(model::GeometricConicForm{T}) where {T}
    empty!(model.dimension)
    fill!(model.num_rows, 0)
    model.A = nothing
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective_constant = zero(T)
end

function MOI.supports_constraint(
    model::GeometricConicForm{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{S}) where {T, S <: MOI.AbstractVectorSet}
    return haskey(model.cone_types_dict, S)
end

function _allocate_variables(model::GeometricConicForm{T, AT, VT}, vis_src, idxmap) where {T, AT, VT}
    model.A = AT(length(vis_src))
    for (i, vi) in enumerate(vis_src)
        idxmap[vi] = MOI.VariableIndex(i)
    end
    return
end

function rows(model::GeometricConicForm{T}, ci::CI{MOI.VectorAffineFunction{T}}) where T
    return ci.value .+ (1:model.dimension[ci.value])
end

function MOI.set(::GeometricConicForm, ::MOI.VariablePrimalStart,
                 ::MOI.VariableIndex, ::Nothing)
end
function MOI.set(model::GeometricConicForm{T}, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::T) where {T}
    model.primal[vi.value] = value
end
function MOI.set(::GeometricConicForm, ::MOI.ConstraintPrimalStart,
                 ::MOI.ConstraintIndex, ::Nothing)
end
function MOI.set(model::GeometricConicForm, ::MOI.ConstraintPrimalStart,
                 ci::MOI.ConstraintIndex, value)
    offset = constroffset(model, ci)
    model.slack[rows(model, ci)] .= value
end
function MOI.set(::GeometricConicForm, ::MOI.ConstraintDualStart,
                  ::MOI.ConstraintIndex, ::Nothing)
end
function MOI.set(model::GeometricConicForm, ::MOI.ConstraintDualStart,
                  ci::MOI.ConstraintIndex, value)
    offset = constroffset(model, ci)
    model.dual[rows(model, ci)] .= value
end
function MOI.set(model::GeometricConicForm, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
end
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
function MOI.set(model::GeometricConicForm{T}, ::MOI.ObjectiveFunction,
                 f::MOI.ScalarAffineFunction{T}) where {T}
    c = Vector(sparsevec(variable_index_value.(f.terms), MOI.coefficient.(f.terms),
                         model.A.n))
    model.objective_constant = f.constant
    model.c = c
    return nothing
end

function _allocate_constraint(model::GeometricConicForm, src, indexmap, cone_id, ci)
    # TODO use `CanonicalConstraintFunction`
    func = MOI.get(src, MOI.ConstraintFunction(), ci)
    func = MOIU.is_canonical(func) ? func : MOI.Utilities.canonical(func)
    allocate_terms(model.A, indexmap, func)
    offset = model.num_rows[cone_id]
    model.num_rows[cone_id] = offset + MOI.output_dimension(func)
    return ci, offset, func
end

function _allocate_constraints(model::GeometricConicForm{T}, src, indexmap, cone_id, ::Type{S}) where {T, S}
    cis = MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T}, S}())
    return map(cis) do ci
        _allocate_constraint(model, src, indexmap, cone_id, ci)
    end
end

function _load_variables(model::GeometricConicForm, nvars::Integer)
    m = sum(model.num_rows)
    model.A.m = m
    model.b = zeros(m)
    model.c = zeros(model.A.n)
    allocate_nonzeros(model.A)
end

function _load_constraints(model::GeometricConicForm, src, indexmap, cone_offset, i, cache)
    for (ci_src, offset_in_cone, func) in cache
        offset = cone_offset + offset_in_cone
        set = MOI.get(src, MOI.ConstraintSet(), ci_src)
        load_terms(model.A, indexmap, func, offset)
        copyto!(model.b, offset + 1, func.constants)
        model.dimension[offset] = MOI.output_dimension(func)
        indexmap[ci_src] = typeof(ci_src)(offset)
    end
end

function MOI.copy_to(dest::GeometricConicForm{T}, src::MOI.ModelLike) where T
    MOI.empty!(dest)

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    idxmap = MOIU.IndexMap()

    has_constraints = BitSet()
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        i = get(dest.cone_types_dict, S, nothing)
        if i === nothing || F != MOI.VectorAffineFunction{T}
            throw(MOI.UnsupportedConstraint{F, S}())
        end
        push!(has_constraints, i)
    end

    _allocate_variables(dest, vis_src, idxmap)

    # Allocate constraints
    caches = map(collect(has_constraints)) do i
        _allocate_constraints(dest, src, idxmap, i, dest.cone_types[i])
    end

    # Load variables
    _load_variables(dest, length(vis_src))

    # Set variable attributes
    MOIU.pass_attributes(dest, src, idxmap, vis_src)

    # Set model attributes
    MOIU.pass_attributes(dest, src, idxmap)

    # Load constraints
    offset = 0
    for (i, cache) in zip(has_constraints, caches)
        _load_constraints(dest, src, idxmap, offset, i, cache)
        offset += dest.num_rows[i]
    end

    final_touch(dest.A)

    return idxmap
end
