mutable struct ConeData
    f::Int # number of linear equality constraints
    l::Int # length of LP cone
    q::Int # length of SOC cone
    qa::Vector{Int} # array of second-order cone constraints
    s::Int # length of SD cone
    sa::Vector{Int} # array of semi-definite constraints
    ep::Int # number of primal exponential cone triples
    ed::Int # number of dual exponential cone triples
    p::Vector{Float64} # array of power cone params
    nrows::Dict{Int, Int} # The number of rows of each vector sets, this is used by `constrrows` to recover the number of rows used by a constraint when getting `ConstraintPrimal` or `ConstraintDual`
    function ConeData()
        new(0, 0, 0, Int[], 0, Int[], 0, 0, Float64[], Dict{Int, Int}())
    end
end

mutable struct ModelData
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    I::Vector{Int} # List of rows
    J::Vector{Int} # List of cols
    V::Vector{Float64} # List of coefficients
    b::Vector{Float64} # constants
    objective_constant::Float64 # The objective is min c'x + objective_constant
    c::Vector{Float64}
end

mutable struct ConicData
    cone::ConeData
    data::Union{Nothing, ModelData}
    function ConicData()
        return new(ConeData(), nothing)
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

cons_offset(cone::MOI.Zeros) = cone.dimension
cons_offset(cone::MOI.Nonnegatives) = cone.dimension
cons_offset(cone::MOI.SecondOrderCone) = cone.dimension
cons_offset(cone::MOI.PositiveSemidefiniteConeTriangle) = Int64((cone.side_dimension*(cone.side_dimension+1))/2)

function restructure_arrays(_s::Array{Float64}, _y::Array{Float64}, cones::Array{<: MOI.AbstractVectorSet})
    i=0
    s = Array{Float64}[]
    y = Array{Float64}[]
    for cone in cones
        offset = cons_offset(cone)
        push!(s, _s[i.+(1:offset)])
        push!(y, _y[i.+(1:offset)])
        i += offset
    end
    return s, y
end

# Computes cone dimensions
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    return ci.value
end
#_allocate_constraint: Allocate indices for the constraint `f`-in-`s`
# using information in `cone` and then update `cone`
function _allocate_constraint(cone::ConeData, f, s::MOI.Zeros)
    ci = cone.f
    cone.f += MOI.dimension(s)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.Nonnegatives})
    return cone.f + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.Nonnegatives)
    ci = cone.l
    cone.l += MOI.dimension(s)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.SecondOrderCone})
    return cone.f + cone.l + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.SecondOrderCone)
    push!(cone.qa, s.dimension)
    ci = cone.q
    cone.q += MOI.dimension(s)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction,
                             MOI.PositiveSemidefiniteConeTriangle})
    return cone.f + cone.l + cone.q + ci.value
end
function _allocate_constraint(cone::ConeData, f,
                              s::MOI.PositiveSemidefiniteConeTriangle)
    push!(cone.sa, s.side_dimension)
    ci = cone.s
    cone.s += MOI.dimension(s)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.ExponentialCone})
    return cone.f + cone.l + cone.q + cone.s + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.ExponentialCone)
    ci = 3cone.ep
    cone.ep += 1
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.DualExponentialCone})
    return cone.f + cone.l + cone.q + cone.s + 3cone.ep + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.DualExponentialCone)
    ci = 3cone.ed
    cone.ed += 1
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, <:MOI.PowerCone})
    return cone.f + cone.l + cone.q + cone.s + 3cone.ep + 3cone.ed + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.PowerCone)
    ci = length(cone.p)
    push!(cone.p, s.exponent)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, <:MOI.DualPowerCone})
    return cone.f + cone.l + cone.q + cone.s + 3cone.ep + 3cone.ed + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.DualPowerCone)
    ci = length(cone.p)
    # SCS' convention: dual cones have a negative exponent.
    push!(cone.p, -s.exponent)
    return ci
end
function constroffset(conic::ConicData, ci::CI)
    return constroffset(conic.cone, ci::CI)
end
function __allocate_constraint(conic::ConicData, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return CI{F, S}(_allocate_constraint(conic.cone, f, s))
end

# Vectorized length for matrix dimension n
sympackedlen(n) = div(n*(n+1), 2)
# Matrix dimension for vectorized length n
sympackeddim(n) = div(isqrt(1+8n) - 1, 2)
trimap(i::Integer, j::Integer) = i < j ? trimap(j, i) : div((i-1)*i, 2) + j
trimapL(i::Integer, j::Integer, n::Integer) = i < j ? trimapL(j, i, n) : i + div((2n-j) * (j-1), 2)
function _sympackedto(x, n, mapfrom, mapto)
    @assert length(x) == sympackedlen(n)
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


# Scale coefficients depending on rows index
# rows: List of row indices
# coef: List of corresponding coefficients
# d: dimension of set
# rev: if true, we unscale instead (e.g. divide by √2 instead of multiply for PSD cone)
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
constrrows(conic::ConicData, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:conic.cone.nrows[constroffset(conic, ci)]

orderval(val, s) = val
function orderval(val, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoL(val, s.side_dimension)
end
orderidx(idx, s) = idx
function orderidx(idx, s::MOI.PositiveSemidefiniteConeTriangle)
    sympackedUtoLidx(idx, s.side_dimension)
end
function __load_constraint(conic::ConicData, ci::MOI.ConstraintIndex, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet)
    func = MOIU.canonical(f)
    I = Int[output_index(term) for term in func.terms]
    J = Int[variable_index_value(term) for term in func.terms]
    V = Float64[-coefficient(term) for term in func.terms]
    offset = constroffset(conic, ci)
    rows = constrrows(s)
    conic.cone.nrows[offset] = length(rows)
    i = offset .+ rows
    b = f.constants
    # @warn ci
    # @warn offset
    # @warn rows
    if s isa MOI.PositiveSemidefiniteConeTriangle
        b = scalecoef(rows, b, s)
        b = sympackedUtoL(b, s.side_dimension)
        V = scalecoef(I, V, s)
        I = sympackedUtoLidx(I, s.side_dimension)
    end
    # The SCS format is b - Ax ∈ cone
    conic.data.b[i] = b
    append!(conic.data.I, offset .+ I)
    append!(conic.data.J, J)
    append!(conic.data.V, V)
end

function __allocate_variables(conic::ConicData, nvars::Integer)
    conic.cone = ConeData()
    VI.(1:nvars)
end

function __load_variables(conic::ConicData, nvars::Integer)
    cone = conic.cone
    m = cone.f + cone.l + cone.q + cone.s + 3cone.ep + 3cone.ed + 3length(cone.p)
    I = Int[]
    J = Int[]
    V = Float64[]
    b = zeros(m)
    c = zeros(nvars)
    conic.data = ModelData(m, nvars, I, J, V, b, 0., c)
end

function __load(conic::ConicData, ::MOI.ObjectiveFunction, f::MOI.ScalarAffineFunction)
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms), conic.data.n))
    conic.data.objective_constant = f.constant
    conic.data.c = c0
end
