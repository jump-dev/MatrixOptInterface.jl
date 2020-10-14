function change_form(::Type{LPForm{T, AT, VT}}, lp::LPForm) where {T, AT, VT}
    return LPForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        lp.c_lb,
        lp.c_ub,
        lp.v_lb,
        lp.v_ub
    )
end

function change_form(::Type{LPForm{T, AT, VT}}, lp::LPStandardForm{T}) where {T, AT, VT}
    return LPForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        lp.b,
        lp.b,
        fill(zero(T), length(lp.c)),
        fill(typemax(T), length(lp.c)),
    )
end
function change_form(::Type{LPForm{T, AT, VT}}, lp::LPGeometricForm{T}) where {T, AT, VT}
    return LPForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        fill(typemin(T), length(lp.b)),
        lp.b,
        fill(typemin(T), length(lp.c)),
        fill(typemax(T), length(lp.c)),
    )
end
function change_form(::Type{LPForm{T, AT, VT}}, lp::LPSolverForm{T}) where {T, AT, VT}
    c_lb = fill(typemin(T), length(lp.b))
    c_ub = fill(typemax(T), length(lp.b))
    for i in eachindex(lp.b)
        if lp.senses[i] == LESS_THAN
            c_ub[i] = lp.b[i]
        elseif lp.senses[i] == GREATER_THAN
            c_lb[i] = lp.b[i]
        elseif lp.senses[i] == EQUAL_TO
            c_lb[i] = lp.b[i]
            c_ub[i] = lp.b[i]
        else
            error("invalid sign $(lp.senses[i])")
        end
    end
    return LPForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        c_lb,
        c_ub,
        lp.v_lb,
        lp.v_ub,
    )
end

function change_form(::Type{LPGeometricForm{T, AT, VT}}, lp::LPGeometricForm) where {T, AT, VT}
    return LPGeometricForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        lp.b
    )
end
function change_form(::Type{LPGeometricForm{T, AT, VT}}, lp::LPForm{T}) where {T, AT, VT}
    has_c_upper = Int[]
    has_c_lower = Int[]
    sizehint!(has_c_upper, length(lp.c_ub))
    sizehint!(has_c_lower, length(lp.c_ub))
    for i in eachindex(lp.c_ub)
        if _no_upper(lp.c_ub[i])
            push!(has_c_upper, i)
        end
        if _no_lower(lp.c_lb[i])
            push!(has_c_lower, i)
        end
    end
    has_v_upper = Int[]
    has_v_lower = Int[]
    sizehint!(has_v_upper, length(lp.v_ub))
    sizehint!(has_v_lower, length(lp.v_ub))
    for i in eachindex(lp.v_ub)
        if _no_upper(lp.v_ub[i])
            push!(has_v_upper, i)
        end
        if _no_lower(lp.v_lb[i])
            push!(has_v_lower, i)
        end
    end
    Id = Matrix{T}(I, length(lp.c), length(lp.c))
    new_A = vcat(
        lp.A[has_c_upper,:],
        -lp.A[has_c_lower,:],
        Id[has_v_upper,:],
        -Id[has_v_lower,:],
                )
    new_b = vcat(
        lp.c_ub[has_c_upper],
        -lp.c_lb[has_c_lower],
        lp.v_ub[has_v_upper],
        -lp.v_lb[has_v_lower],
    )
    return LPGeometricForm{T, AT, VT}(
        lp.direction,
        lp.c,
        new_A,
        new_b
    )
end
function change_form(::Type{LPGeometricForm{T, AT, VT}}, lp::F) where {T, AT, VT, F <: AbstractLPForm{T}}
    temp_lp = change_form(LPForm{T, AT, VT}, lp)
    return change_form(LPGeometricForm{T, AT, VT}, temp_lp)
end

function change_form(::Type{LPStandardForm{T, AT, VT}}, lp::LPStandardForm) where {T, AT, VT}
    return LPStandardForm(
        lp.direction,
        lp.c,
        lp.A,
        lp.b
    )
end
function change_form(::Type{LPStandardForm{T, AT, VT}}, lp::LPGeometricForm{T}) where {T, AT, VT}
    new_A = hcat(
        lp.A,
        -lp.A,
        AT(I, length(lp.b), length(lp.b))
    )
    new_c = vcat(
        lp.c,
        -lp.c,
        fill(0.0, length(lp.b))
    )
    return LPStandardForm{T, AT, VT}(
        lp.direction,
        new_c,
        new_A,
        copy(lp.b)
    )
end
function change_form(::Type{LPStandardForm{T, AT, VT}}, lp::F) where {T, AT, VT, F <: AbstractLPForm{T}}
    temp_lp = change_form(LPForm{T, AT, VT}, lp)
    new_lp = change_form(LPGeometricForm{T, AT, VT}, temp_lp)
    change_form(LPStandardForm{T, AT, VT}, new_lp)
end

function change_form(::Type{LPSolverForm{T, AT, VT}}, lp::LPSolverForm) where {T, AT, VT}
    return LPSolverForm{T, AT, VT}(
        lp.direction,
        lp.c,
        lp.A,
        lp.b,
        lp.senses,
        lp.v_lb,
        lp.v_ub
    )
end
function change_form(::Type{LPSolverForm{T, AT, VT}}, lp::LPForm{T}) where {T, AT, VT}
    new_A = copy(lp.A)
    senses = fill(LESS_THAN, length(lp.c_lb))
    new_b = fill(NaN, length(lp.c_lb))
    for i in eachindex(lp.c_lb)
        if lp.c_lb[i] == lp.c_ub[i]
            senses[i] = EQUAL_TO
            new_b[i] = lp.c_lb[i]
        elseif lp.c_lb[i] > -Inf && lp.c_ub[i] < Inf
            senses[i] = GREATER_THAN
            new_b[i] = lp.c_lb[i]
            push!(new_b, lp.c_ub[i])
            push!(sense, LESS_THAN)
            new_A = vcat(new_A, lp.A[i,:])
        elseif lp.c_lb[i] > -Inf
            senses[i] = GREATER_THAN
            new_b[i] = lp.c_lb[i]
        elseif lp.c_ub[i] < Inf
            senses[i] = LESS_THAN
            new_b[i] = lp.c_ub[i]
        end
    end
    return LPSolverForm{T, AT, VT}(
        lp.direction,
        lp.c,
        new_A,
        new_b,
        senses,
        lp.v_lb,
        lp.v_ub,
    )
end
function change_form(::Type{LPSolverForm{T, AT, VT}}, lp::F) where {T, AT, VT, F <: AbstractLPForm{T}}
    temp_lp = change_form(LPForm{T, AT, VT}, lp)
    change_form(LPSolverForm{T, AT, VT}, temp_lp)
end
