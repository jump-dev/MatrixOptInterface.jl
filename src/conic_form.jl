mutable struct ModelData{T}
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    I::Vector{Int} # List of rows
    J::Vector{Int} # List of cols
    V::Vector{T} # List of coefficients
    b::Vector{T} # constants
    objective_constant::T # The objective is min c'x + objective_constant
    c::Vector{T}
end

mutable struct ConicData{T}
    f::Int # number of linear equality constraints
    l::Int # length of LP conic
    q::Int # length of SOC conic
    qa::Vector{Int} # array of second-order conic constraints
    s::Int # length of SD conic
    sa::Vector{Int} # array of semi-definite constraints
    ep::Int # number of primal exponential conic triples
    ed::Int # number of dual exponential conic triples
    p::Vector{Float64} # array of power conic params
    nrows::Dict{Int, Int} # The number of rows of each vector sets, this is used by `constrrows` to recover the number of rows used by a constraint when getting `ConstraintPrimal` or `ConstraintDual`
    data::Union{Nothing, ModelData{T}}

    function ConicData{T}() where T
        return new{T}(0, 0, 0, Int[], 0, Int[], 0, 0, Float64[], Dict{Int, Int}(), nothing)
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

const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

cons_offset(conic::MOI.Zeros) = conic.dimension
cons_offset(conic::MOI.Nonnegatives) = conic.dimension
cons_offset(conic::MOI.SecondOrderCone) = conic.dimension
cons_offset(conic::MOI.PositiveSemidefiniteConeTriangle) = Int64((conic.side_dimension*(conic.side_dimension+1))/2)

function restructure_arrays(_s::Array{T}, _y::Array{T}, cones::Array{<: MOI.AbstractVectorSet}) where {T}
    i=0
    s = Array{T}[]
    y = Array{T}[]
    for conic in cones
        offset = cons_offset(conic)
        push!(s, _s[i.+(1:offset)])
        push!(y, _y[i.+(1:offset)])
        i += offset
    end
    return s, y
end

# Computes conic dimensions
function constraint_offset(conic::ConicData{T}, ci::CI{<:MOI.AbstractFunction, MOI.Zeros}) where T
    return ci.value
end

"""
    _allocate_constraint
Allocate indices for the constraint `f`-in-`s` using information in `conic` and then update `conic`.
"""
function _allocate_constraint(conic::ConicData{T}, f, s::MOI.Zeros) where T
    ci = conic.f
    conic.f += MOI.dimension(s)
    return ci
end
function constraint_offset(conic::ConicData{T},ci::CI{<:MOI.AbstractFunction, MOI.Nonnegatives}) where T
    return conic.f + ci.value
end
function _allocate_constraint(conic::ConicData{T}, f, s::MOI.Nonnegatives) where T
    ci = conic.l
    conic.l += MOI.dimension(s)
    return ci
end
function constraint_offset(conic::ConicData{T},
                      ci::CI{<:MOI.AbstractFunction, MOI.SecondOrderCone}) where T
    return conic.f + conic.l + ci.value
end
function _allocate_constraint(conic::ConicData{T}, f, s::MOI.SecondOrderCone) where T
    push!(conic.qa, s.dimension)
    ci = conic.q
    conic.q += MOI.dimension(s)
    return ci
end
function constraint_offset(conic::ConicData{T},
                      ci::CI{<:MOI.AbstractFunction,
                             MOI.PositiveSemidefiniteConeTriangle}) where T
    return conic.f + conic.l + conic.q + ci.value
end
function _allocate_constraint(conic::ConicData{T}, f,
                              s::MOI.PositiveSemidefiniteConeTriangle) where T
    push!(conic.sa, s.side_dimension)
    ci = conic.s
    conic.s += MOI.dimension(s)
    return ci
end
# function constraint_offset(conic::ConicData{T},
#                       ci::CI{<:MOI.AbstractFunction, MOI.ExponentialCone}) where T
#     return conic.f + conic.l + conic.q + conic.s + ci.value
# end
# function _allocate_constraint(conic::ConicData{T}, f, s::MOI.ExponentialCone) where T
#     ci = 3 * conic.ep
#     conic.ep += 1
#     return ci
# end
# function constraint_offset(conic::ConicData{T},
#                       ci::CI{<:MOI.AbstractFunction, MOI.DualExponentialCone}) where T
#     return conic.f + conic.l + conic.q + conic.s + 3 * conic.ep + ci.value
# end
# function _allocate_constraint(conic::ConicData{T}, f, s::MOI.DualExponentialCone) where T
#     ci = 3 * conic.ed
#     conic.ed += 1
#     return ci
# end
# function constraint_offset(conic::ConicData{T},
#                       ci::CI{<:MOI.AbstractFunction, <:MOI.PowerCone}) where T
#     return conic.f + conic.l + conic.q + conic.s + 3 * conic.ep + 3 * conic.ed + ci.value
# end
# function _allocate_constraint(conic::ConicData{T}, f, s::MOI.PowerCone) where T
#     ci = length(conic.p)
#     push!(conic.p, s.exponent)
#     return ci
# end
# function constraint_offset(conic::ConicData{T},
#                       ci::CI{<:MOI.AbstractFunction, <:MOI.DualPowerCone}) where T
#     return conic.f + conic.l + conic.q + conic.s + 3 * conic.ep + 3 * conic.ed + ci.value
# end
# function _allocate_constraint(conic::ConicData{T}, f, s::MOI.DualPowerCone) where T
#     ci = length(conic.p)
#     # SCS' convention: dual cones have a negative exponent.
#     push!(conic.p, -s.exponent)
#     return ci
# end
function __allocate_constraint(conic::ConicData{T}, f::F, s::S) where {T, F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return CI{F, S}(_allocate_constraint(conic, f, s))
end

# Vectorized length for matrix dimension n
sympackedlen(n) = div(n*(n+1), 2)
# Matrix dimension for vectorized length n
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
trimap(i::Integer, j::Integer) = i < j ? trimap(j, i) : div((i-1)*i, 2) + j
trimapL(i::Integer, j::Integer, n::Integer) = i < j ? trimapL(j, i, n) : i + div((2n-j) * (j-1), 2)
function _sympackedto(x, n, mapfrom, mapto)
    length(x) == sympackedlen(n) || throw(DimensionMismatch("error message on dimensions"))
    y = similar(x)
    for i in 1:n, j in 1:i
        y[mapto(i, j)] = x[mapfrom(i, j)]
    end
    y
end
sympackedLtoU(x, n=sympackeddim(length(x))) = _sympackedto(x, n, (i, j) -> trimapL(i, j, n), trimap)
sympackedUtoL(x, n=sympackeddim(length(x))) = _sympackedto(x, n, trimap, (i, j) -> trimapL(i, j, n))

function sympackedUtoLidx(x::AbstractVector{<:Integer}, n)
    y = similar(x)
    map = sympackedLtoU(1:sympackedlen(n), n)
    for i in eachindex(y)
        y[i] = map[x[i]]
    end
    y
end

"""
    _scalecoef(rows::AbstractVector{<: Integer}, coef::Vector{Float64}, d::Integer, rev::Bool)

Scale coefficients depending on rows index
rows: List of row indices
coef: List of corresponding coefficients
d: dimension of set
rev: if true, we unscale instead (e.g. divide by √2 instead of multiply for PSD conic)
"""
function _scalecoef(rows::AbstractVector{<: Integer}, coef::Vector{Float64}, d::Integer, rev::Bool)
    scaling = rev ? 1 / √2 : 1 * √2
    output = copy(coef)
    for i in 1:length(output)
        # See https://en.wikipedia.org/wiki/Triangular_number#Triangular_roots_and_tests_for_triangular_numbers
        val = 8 * rows[i] + 1
        is_diagonal_index = isqrt(val)^2 == val
        if !is_diagonal_index
            output[i] *= scaling
        end
    end
    return output
end

# Unscale the coefficients in `coef` with respective rows in `rows` for a set `s`
scalecoef(rows, coef, s) = _scalecoef(rows, coef, MOI.dimension(s), false)
# Unscale the coefficients in `coef` with respective rows in `rows` for a set of type `S` with dimension `d`
unscalecoef(rows, coef, d) = _scalecoef(rows, coef, sympackeddim(d), true)

output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
# constrrows: Recover the number of rows used by each constraint.
# When, the set is available, simply use MOI.dimension
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
# When only the index is available, use the `conic.ncone.nrows` field
constrrows(conic::ConicData{T}, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) where T = 1:conic.nrows[constraint_offset(conic, ci)]

orderval(val, s) = val
function orderval(val, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoL(val, s.side_dimension)
end
orderidx(idx, s) = idx
function orderidx(idx, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoLidx(idx, s.side_dimension)
end
function __load_constraint(conic::ConicData{T}, ci::MOI.ConstraintIndex, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet) where {T}
    func = MOIU.canonical(f)
    I = Int[output_index(term) for term in func.terms]
    J = Int[variable_index_value(term) for term in func.terms]
    V = T[-coefficient(term) for term in func.terms]
    offset = constraint_offset(conic, ci)
    rows = constrrows(s)
    conic.nrows[offset] = length(rows)
    i = offset .+ rows
    b = f.constants
    if s isa MOI.PositiveSemidefiniteConeTriangle
        b = scalecoef(rows, b, s)
        b = sympackedUtoL(b, s.side_dimension)
        V = scalecoef(I, V, s)
        I = sympackedUtoLidx(I, s.side_dimension)
    end
    # The SCS format is b - Ax ∈ conic
    conic.data.b[i] = b
    append!(conic.data.I, offset .+ I)
    append!(conic.data.J, J)
    append!(conic.data.V, V)
end

function __load_variables(conic::ConicData{T}, nvars::Integer) where T
    m = conic.f + conic.l + conic.q + conic.s + 3 * conic.ep + 3 * conic.ed + 3 * length(conic.p)
    I = Int[]
    J = Int[]
    V = zeros(T, 0)
    b = zeros(T, m)
    c = zeros(T, nvars)
    conic.data = ModelData(m, nvars, I, J, V, b, zero(T), c)
end

function __load(conic::ConicData{T}, ::MOI.ObjectiveFunction, f::MOI.ScalarAffineFunction) where T
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms), conic.data.n))
    conic.data.objective_constant = f.constant
    conic.data.c = c0
end
