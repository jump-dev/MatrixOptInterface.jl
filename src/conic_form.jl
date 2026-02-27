# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    empty_geometric_conic_form

Represents an optimization model of the form:
```
sense ⟨c, x⟩ + c0
s.t.  b_i - A_i x ∈ C_i ∀ i
```
with each `C_i` a cone defined in MOI.
"""
function empty_geometric_conic_form(cones; Tv = Float64, Ti = Int, I = MOI.Utilities.OneBasedIndexing)
    return MOI.Utilities.GenericModel{T}(
        MOI.Utilities.ObjectiveContainer{T}(),
        MOI.Utilities.FreeVariables(),
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                Tv,
                Ti,
                I,
            },
            Vector{T},
            ProductOfSets{T},
        },
    )
end

function geometric_conic_form(model::MOI.ModelLike, cones; kws...)
    form = empty_geometric_conic_form(cones; kws...)
    index_map = MOI.copy_to(form, model)
    return form, index_map
end

_coef_type(::MOI.Utilities.AbstractModel{T}) where {T} = T

function objective_vector(model::MOI.ModelLike; T = _coef_type(model))
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
    dest.objective_constant = MOI.constant(obj)
    c = zeros(A.n)
    for term in obj.terms
        c[term.variable.value] += term.coefficient
    end
    return c
end
