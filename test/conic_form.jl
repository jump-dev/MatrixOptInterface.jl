# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function _test_matrix_equal(A::SparseMatrixCSC, B::SparseMatrixCSC)
    @test A.m == B.m
    @test A.n == B.n
    @test A.nzval ≈ B.nzval atol = ATOL rtol = RTOL
    @test A.rowval == B.rowval
    @test A.colptr == B.colptr
end

# _psd1test: https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L2417
function psd1(::Type{T}, ::Type{I}) where {T,I}
    # We use `MockOptimizer` to have indices xor'ed so that it tests that we don't assumes they are `1:n`.
    model = MOIU.MockOptimizer(MOIU.Model{T}())

    X = MOI.add_variables(model, 6)
    x = MOI.add_variables(model, 3)

    vov = MOI.VectorOfVariables(X)

    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{T}(vov),
        MOI.PositiveSemidefiniteConeTriangle(3),
    )

    cx = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{T}(MOI.VectorOfVariables(x)),
        MOI.SecondOrderCone(3),
    )

    c1 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1:1,
                MOI.ScalarAffineTerm.(ones(T, 4), [X[1], X[3], X[end], x[1]]),
            ),
            [-one(T)],
        ),
        MOI.Zeros(1),
    )

    c2 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1:1,
                MOI.ScalarAffineTerm.(
                    T[1, 2, 1, 2, 2, 1, 1, 1],
                    [X; x[2]; x[3]],
                ),
            ),
            [-inv(T(2))],
        ),
        MOI.Zeros(1),
    )

    objXidx = [1:3; 5:6]
    objXcoefs = 2ones(T, 5)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([objXcoefs; one(T)], [X[objXidx]; x[1]]),
            zero(T),
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    conic_form, index_map = MatOI.geometric_conic_form(
        model,
        [
            MOI.PositiveSemidefiniteConeTriangle,
            MOI.SecondOrderCone,
            MOI.Zeros,
        ];
        Tv = T,
        I,
    )
    @test index_map isa MOI.Utilities.IndexMap

    @test MatOI.objective_vector(conic_form)' ≈ T[2 2 2 0 2 2 1 0 0]
    @test conic_form.constraints.constants' ≈ T[0 0 0 0 0 0 0 0 0 -1 -inv(T(2))]
    return _test_matrix_equal(
        conic_form.constraints.coefficients,
        SparseMatrixCSC(
            11,
            9,
            [1, 4, 6, 9, 11, 13, 16, 18, 20, 22],
            [
                1,
                10,
                11,
                2,
                11,
                3,
                10,
                11,
                4,
                11,
                5,
                11,
                6,
                10,
                11,
                7,
                10,
                8,
                11,
                9,
                11,
            ],
            T[
                -1,
                -1,
                -1,
                -1,
                -2,
                -1,
                -1,
                -1,
                -1,
                -2,
                -1,
                -2,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
        ),
    )
end

# Taken from `MOI.Test.psdt2test`.
# find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
function psd2(
    ::Type{T},
    ::Type{I},
    η::T = T(10),
    α::T = T(4) / T(5),
    δ::T = T(9) / T(10),
) where {T,I}
    # We use `MockOptimizer` to have indices xor'ed so that it tests that we don't assumes they are `1:n`.
    model = MOIU.MockOptimizer(MOIU.Model{T}())

    x = MOI.add_variables(model, 7)

    c1 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(-one(T), x[1:6])),
            [η],
        ),
        MOI.Nonnegatives(1),
    )
    c2 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1:6, MOI.ScalarAffineTerm.(one(T), x[1:6])),
            zeros(T, 6),
        ),
        MOI.Nonnegatives(6),
    )

    c3 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                [fill(1, 7); fill(2, 5); fill(3, 6)],
                MOI.ScalarAffineTerm.(
                    [
                        δ / 2,
                        α,
                        δ,
                        δ / 4,
                        δ / 8,
                        0,
                        -1,
                        -δ / (2 * √2),
                        -δ / 4,
                        0,
                        -δ / (8 * √2),
                        0,
                        δ / 2,
                        δ - α,
                        0,
                        δ / 8,
                        δ / 4,
                        -1,
                    ],
                    [x[1:7]; x[1:3]; x[5:6]; x[1:3]; x[5:7]],
                ),
            ),
            zeros(T, 3),
        ),
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    c4 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1,
                MOI.ScalarAffineTerm.(zero(T), [x[1:3]; x[5:6]]),
            ),
            [zero(T)],
        ),
        MOI.Zeros(1),
    )

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(one(T), x[7])], zero(T)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    conic_form, index_map = MatOI.geometric_conic_form(
        model,
        [MOI.Nonnegatives, MOI.Zeros, MOI.PositiveSemidefiniteConeTriangle];
        Tv = T,
        I,
    )
    @test index_map isa MOI.Utilities.IndexMap

    @test MatOI.objective_vector(conic_form) ≈ [zeros(T, 6); one(T)]
    @test conic_form.constraint.constants ≈ [T(10); zeros(T, 10)]
    return _test_matrix_equal(
        conic_form.constraint.coefficients,
        SparseMatrixCSC(
            11,
            7,
            [1, 6, 11, 14, 17, 22, 25, 27],
            [
                1,
                2,
                9,
                10,
                11,
                1,
                3,
                9,
                10,
                11,
                1,
                4,
                9,
                1,
                5,
                9,
                1,
                6,
                9,
                10,
                11,
                1,
                7,
                11,
                9,
                11,
            ],
            T[
                1.0,
                -1.0,
                -0.45,
                0.318198,
                -0.45,
                1.0,
                -1.0,
                -0.8,
                0.225,
                -0.1,
                1.0,
                -1.0,
                -0.9,
                1.0,
                -1.0,
                -0.225,
                1.0,
                -1.0,
                -0.1125,
                0.0795495,
                -0.1125,
                1.0,
                -1.0,
                -0.225,
                1.0,
                1.0,
            ],
        ),
    )
end

@testset "PSD $T, $I" for T in [Float64, BigFloat],
    I in [MOI.Utilities.ZeroBasedIndexing, MOI.Utilities.OneBasedIndexing]

    psd1(T, I)
    psd2(T, I)
end
