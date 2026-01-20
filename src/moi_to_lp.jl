import MathOptInterface

const MOI = MathOptInterface

MOI.Utilities.@product_of_sets(
    _LPProductOfSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
)

const LinearOptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Float64},
        _LPProductOfSets{Float64},
    },
}

const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

# =======================
#   `copy_to` function
# =======================

function _index_map(
    src::LinearOptimizerCache,
    index_map,
    ::Type{F},
    ::Type{S},
) where {F,S}
    inner = index_map.con_map[F, S]
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        row = MOI.Utilities.rows(src.constraints, ci)
        inner[ci] = MOI.ConstraintIndex{F,S}(row)
    end
    return
end

function _index_map(
    src::LinearOptimizerCache,
    index_map,
    F::Type{MOI.VariableIndex},
    ::Type{S},
) where {S}
    inner = index_map.con_map[F, S]
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        col = index_map[MOI.VariableIndex(ci.value)].value
        inner[ci] = MOI.ConstraintIndex{F,S}(col)
    end
    return
end

"""
    _index_map(src::LinearOptimizerCache)
Create an `IndexMap` mapping the variables and constraints in `LinearOptimizerCache`
to their corresponding 1-based columns and rows.
"""
function _index_map(src::LinearOptimizerCache)
    index_map = MOI.IndexMap()
    for (i, x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        index_map[x] = MOI.VariableIndex(i)
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        _index_map(src, index_map, F, S)
    end
    return index_map
end

function MOI.copy_to(dest::LPForm{T,AT,VT}, src::LinearOptimizerCache) where {T,AT,VT}
    obj =
        MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    c = zeros(length(src.variables.lower))
    for term in obj.terms
        c[term.variable.value] += term.coefficient
    end
    # handle constant obj?
    # obj.constant
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    dest.c = convert(VT, c)
    dest.A = convert(AT, src.constraints.coefficients)
    dest.c_lb = convert(VT, src.constraints.constants.lower)
    dest.c_ub = convert(VT, src.constraints.constants.upper)
    dest.v_lb = convert(VT, src.variables.lower)
    dest.v_ub = convert(VT, src.variables.upper)
    map = _index_map(src)
    return map
end

function MOI.copy_to(dest::LPForm{T,AT,VT}, src::MOI.ModelLike) where {T,AT,VT}
    # check supported constraints
    cache = LinearOptimizerCache()
    src_to_cache = MOI.copy_to(cache, src)
    cache_to_dest = MOI.copy_to(dest, cache)
    index_map = MOI.IndexMap()
    for (src_x, cache_x) in src_to_cache.var_map
        index_map[src_x] = cache_to_dest[cache_x]
    end
    for (src_ci, cache_ci) in src_to_cache.con_map
        index_map[src_ci] = cache_to_dest[cache_ci]
    end
    return index_map
end
