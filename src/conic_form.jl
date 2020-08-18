mutable struct ConicForm{T, AT, VT, C} <: MOI.ModelLike
    num_rows::Vector{Int}
    dimension::Dict{Int, Int}
    sense::MOI.OptimizationSense
    objective_constant::T # The objective
    A::Union{Nothing, AT} # The constraints
    b::VT          # `b - Ax in cones`
    c::VT          # `sense c'x + objective_constant`
    cone_types::C

    function ConicForm{T, AT, VT}(cone_types) where {T, AT, VT}
        model = new{T, AT, VT, typeof(cone_types)}()
        model.cone_types = cone_types
        model.num_rows = zeros(Int, length(cone_types))
        model.dimension = Dict{Int, Int}()
        model.A = nothing
        return model
    end
end

# The ordering of CONES matches SCS
const CONES = Dict(
    MOI.Zeros => 1,
    MOI.Nonnegatives => 2,
    MOI.SecondOrderCone => 3,
    MOI.PositiveSemidefiniteConeTriangle => 4,
    MOI.ExponentialCone => 5, 
    MOI.DualExponentialCone => 6
)

cons_offset(conic::MOI.Zeros) = conic.dimension
cons_offset(conic::MOI.Nonnegatives) = conic.dimension
cons_offset(conic::MOI.SecondOrderCone) = conic.dimension
cons_offset(conic::MOI.PositiveSemidefiniteConeTriangle) = Int64((conic.side_dimension*(conic.side_dimension+1))/2)

# function restructure_arrays(_s::Array{T}, _y::Array{T}, cones::Array{<: MOI.AbstractVectorSet}) where {T}
#     i=0
#     s = Array{T}[]
#     y = Array{T}[]
#     for conic in cones
#         offset = cons_offset(conic)
#         push!(s, _s[i.+(1:offset)])
#         push!(y, _y[i.+(1:offset)])
#         i += offset
#     end
#     return s, y
# end

function get_conic_form(::Type{T}, model::M, con_idx) where {T, M <: MOI.AbstractOptimizer}
    # reorder constraints
    cis = sort(
        con_idx, 
        by = x->CONES[typeof(MOI.get(model, MOI.ConstraintSet(), x))]
    )

    # extract cones
    cones = typeof.(MOI.get(model, MOI.ConstraintSet(), cis))

    conic = ConicForm{T, SparseMatrixCSRtoCSC{Int64}, Array{T, 1}}(Tuple(cones))

    idxmap = MOI.copy_to(conic, model)

    # fix optimization sense
    if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        conic.c = -conic.c
    end

    return conic
end

MOI.is_empty(model::ConicForm) = model.A === nothing
function MOI.empty!(model::ConicForm{T}) where T
    empty!(model.dimension)
    fill!(model.num_rows, 0)
    model.A = nothing
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective_constant = zero(T)
end

function _first(::Type{S}, idx::Int, ::Type{S}, args::Vararg{Type, N}) where {S, N}
    return idx
end
function _first(::Type{S}, idx::Int, ::Type, args::Vararg{Type, N}) where {S, N}
    return _first(S, idx + 1, args...)
end
_first(::Type, idx::Int) = nothing

_findfirst(::Type{S}, sets::Tuple) where {S} = _first(S, 1, sets...)

function MOI.supports_constraint(
    model::ConicForm,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{S}) where S <: MOI.AbstractVectorSet
    return _findfirst(S, model.cone_types) !== nothing
end

function _allocate_variables(model::ConicForm{T, AT, VT}, vis_src, idxmap) where {T, AT, VT}
    model.A = AT(length(vis_src))
    for (i, vi) in enumerate(vis_src)
        idxmap[vi] = MOI.VariableIndex(i)
    end
    return
end

function rows(model::ConicForm, ci::CI{MOI.VectorAffineFunction{Float64}})
    return ci.value .+ (1:model.dimension[ci.value])
end

function MOI.set(::ConicForm, ::MOI.VariablePrimalStart,
                 ::MOI.VariableIndex, ::Nothing)
end
function MOI.set(model::ConicForm, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::Float64)
    model.primal[vi.value] = value
end
function MOI.set(::ConicForm, ::MOI.ConstraintPrimalStart,
                 ::MOI.ConstraintIndex, ::Nothing)
end
function MOI.set(model::ConicForm, ::MOI.ConstraintPrimalStart,
                 ci::MOI.ConstraintIndex, value)
    offset = constroffset(model, ci)
    model.slack[rows(model, ci)] .= value
end
function MOI.set(::ConicForm, ::MOI.ConstraintDualStart,
                  ::MOI.ConstraintIndex, ::Nothing)
end
function MOI.set(model::ConicForm, ::MOI.ConstraintDualStart,
                  ci::MOI.ConstraintIndex, value)
    offset = constroffset(model, ci)
    model.dual[rows(model, ci)] .= value
end
function MOI.set(model::ConicForm, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
end
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
function MOI.set(model::ConicForm, ::MOI.ObjectiveFunction,
                 f::MOI.ScalarAffineFunction{Float64})
    c = Vector(sparsevec(variable_index_value.(f.terms), MOI.coefficient.(f.terms),
                         model.A.n))
    model.objective_constant = f.constant
    model.c = c
    return nothing
end

function _allocate_constraint(model::ConicForm, src, indexmap, cone_id, ci)
    # TODO use `CanonicalConstraintFunction`
    func = MOI.get(src, MOI.ConstraintFunction(), ci)
    func = MOIU.is_canonical(func) ? func : MOI.Utilities.canonical(func)
    allocate_terms(model.A, indexmap, func)
    offset = model.num_rows[cone_id]
    model.num_rows[cone_id] = offset + MOI.output_dimension(func)
    return ci, offset, func
end

function _allocate_constraints(model::ConicForm, src, indexmap, cone_id, ::Type{S}) where S
    cis = MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64}, S}())
    return map(cis) do ci
        _allocate_constraint(model, src, indexmap, cone_id, ci)
    end
end

function _load_variables(model::ConicForm, nvars::Integer)
    m = sum(model.num_rows)
    model.A.m = m
    model.b = zeros(m)
    model.c = zeros(model.A.n)
    allocate_nonzeros(model.A)
end

function _load_constraints(model::ConicForm, src, indexmap, cone_offset, i, cache, preprocess)
    for (ci_src, offset_in_cone, func) in cache
        offset = cone_offset + offset_in_cone
        set = MOI.get(src, MOI.ConstraintSet(), ci_src)
        new_func = preprocess(func, set)
        load_terms(model.A, indexmap, new_func, offset)
        copyto!(model.b, offset + 1, new_func.constants)
        model.dimension[offset] = MOI.output_dimension(func)
        indexmap[ci_src] = typeof(ci_src)(offset)
    end
end

# Vectorized length for matrix dimension n
sympackedlen(n) = div(n*(n+1), 2)
# Matrix dimension for vectorized length n
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
function _sympackedto(x, n, mapfrom, mapto)
    @assert length(x) == sympackedlen(n)
    y = similar(x)
    for i in 1:n, j in 1:i
        y[mapto(i, j)] = x[mapfrom(i, j)]
    end
    y
end
trimap(i::Integer, j::Integer) = i < j ? trimap(j, i) : div((i-1)*i, 2) + j
trimapL(i::Integer, j::Integer, n::Integer) = i < j ? trimapL(j, i, n) : i + div((2n-j) * (j-1), 2)
sympackedLtoU(x, n=sympackeddim(length(x))) = _sympackedto(x, n, (i, j) -> trimapL(i, j, n), trimap)
sympackedUtoL(x, n) = _sympackedto(x, n, trimap, (i, j) -> trimapL(i, j, n))

function _scale(i, coef)
    if MOI.Utilities.is_diagonal_vectorized_index(i)
        return coef
    else
        return coef * √2
    end
end

function _preprocess_function(func, set::MOI.PositiveSemidefiniteConeTriangle)
    n = set.side_dimension
    LtoU_map = sympackedLtoU(1:sympackedlen(n), n)
    function map_term(t::MOI.VectorAffineTerm)
        return MOI.VectorAffineTerm(
            LtoU_map[t.output_index],
            MOI.ScalarAffineTerm(
                _scale(t.output_index, t.scalar_term.coefficient),
                t.scalar_term.variable_index
            )
        )
    end
    UtoL_map = sympackedUtoL(1:sympackedlen(n), n)
    function constant(row)
        i = UtoL_map[row]
        return _scale(i, func.constants[i])
    end
    new_func = MOI.VectorAffineFunction{Float64}(
        MOI.VectorAffineTerm{Float64}[map_term(t) for t in func.terms],
        constant.(eachindex(func.constants))
    )
    # The rows have been reordered in `map_term` so we need to re-canonicalize to reorder the rows.
    MOI.Utilities.canonicalize!(new_func)
    return new_func
end
_preprocess_function(func, set) = func

function MOI.copy_to(dest::ConicForm, src::MOI.ModelLike; preprocess = _preprocess_function, copy_names::Bool=true)
    MOI.empty!(dest)

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    idxmap = MOIU.IndexMap()

    has_constraints = BitSet()
    for (F, S) in MOI.get(src, MOI.ListOfConstraints())
        i = _findfirst(S, dest.cone_types)
        if i === nothing || F != MOI.VectorAffineFunction{Float64}
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
    MOIU.pass_attributes(dest, src, copy_names, idxmap, vis_src)

    # Set model attributes
    MOIU.pass_attributes(dest, src, copy_names, idxmap)

    # Load constraints
    offset = 0
    for (i, cache) in zip(has_constraints, caches)
        _load_constraints(dest, src, idxmap, offset, i, cache, preprocess)
        offset += dest.num_rows[i]
    end

    return idxmap
end
