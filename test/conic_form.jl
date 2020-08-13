CONIC_OPTIMIZERS = [SCS.Optimizer, ProxSDP.Optimizer, COSMO.Optimizer]

@testset "MOI to MatOI conversion 1" begin
    # _psd1test: https://github.com/jump-dev/MOI.jl/blob/master/src/Test/contconic.jl#L2417

    for optimizer in CONIC_OPTIMIZERS
        model = MOI.instantiate(optimizer, with_bridge_type=Float64)
        δ = √(1 + (3*√2+2)*√(-116*√2+166) / 14) / 2
        ε = √((1 - 2*(√2-1)*δ^2) / (2-√2))
        y2 = 1 - ε*δ
        y1 = 1 - √2*y2
        obj = y1 + y2/2
        k = -2*δ/ε
        x2 = ((3-2obj)*(2+k^2)-4) / (4*(2+k^2)-4*√2)
        α = √(3-2obj-4x2)/2
        β = k*α

        X = MOI.add_variables(model, 6)
        x = MOI.add_variables(model, 3)

        vov = MOI.VectorOfVariables(X)

        cX = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(3)
        )

        cx = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x)), MOI.SecondOrderCone(3)
        )

        c1 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 1., 1., 1.], [X[1], X[3], X[end], x[1]])), 
                [-1.0]
            ), 
            MOI.Zeros(1)
        )

        c2 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 2, 1, 2, 2, 1, 1, 1], [X; x[2]; x[3]])), 
                [-0.5]
            ), 
            MOI.Zeros(1)
        )

        objXidx = [1:3; 5:6]
        objXcoefs = 2*ones(5)
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([objXcoefs; 1.0], [X[objXidx]; x[1]]), 0.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        MatModel = MatOI.get_conic_form(Float64, model, [cX; cx; c1; c2])

        @test MatModel.c' ≈ [2. 2. 2. 0. 2. 2. 1. 0. 0.]
        @test MatModel.b' ≈ [-1.  -0.5  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
        A = MatModel.A
        @test A[1, 1] ≈	-1.0 
        @test A[2, 1] ≈	-1.0
        @test A[6, 1] ≈	-1.0
        @test A[2, 2] ≈	-2.0
        @test A[7, 2] ≈	-1.41421 atol=ATOL rtol=RTOL
        @test A[1, 3] ≈	-1.0
        @test A[2, 3] ≈	-1.0
        @test A[9, 3] ≈	-1.0
        @test A[2, 4] ≈	-2.0
        @test A[8, 4] ≈	-1.41421 atol=ATOL rtol=RTOL
        @test A[2, 5] ≈	-2.0
        @test A[10, 5] ≈ -1.41421 atol=ATOL rtol=RTOL
        @test A[1, 6] ≈	-1.0
        @test A[2, 6] ≈	-1.0
        @test A[11, 6] ≈ -1.0
        @test A[1, 7] ≈	-1.0
        @test A[3, 7] ≈	-1.0
        @test A[2, 8] ≈	-1.0
        @test A[4, 8] ≈	-1.0
        @test A[2, 9] ≈	-1.0
        @test A[5, 9] ≈	-1.0
    end
end

@testset "MOI to MatOI conversion 2" begin
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    
    for optimizer in CONIC_OPTIMIZERS
        model = MOI.instantiate(optimizer, with_bridge_type=Float64)

        x = MOI.add_variables(model, 7)
        @test MOI.get(model, MOI.NumberOfVariables()) == 7

        η = 10.0

        c1  = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(-1.0, x[1:6])),
                [η]
            ), 
            MOI.Nonnegatives(1)
        )
        c2 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.(1:6, MOI.ScalarAffineTerm.(1.0, x[1:6])), zeros(6)), MOI.Nonnegatives(6))
        α = 0.8
        δ = 0.9
        c3 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.([fill(1, 7); fill(2, 5);     fill(3, 6)],
                                                                MOI.ScalarAffineTerm.(
                                                                [ δ/2,       α,   δ, δ/4, δ/8,      0.0, -1.0,
                                                                    -δ/(2*√2), -δ/4, 0,     -δ/(8*√2), 0.0,
                                                                    δ/2,     δ-α,   0,      δ/8,      δ/4, -1.0],
                                                                [x[1:7];     x[1:3]; x[5:6]; x[1:3]; x[5:7]])),
                                                                zeros(3)), MOI.PositiveSemidefiniteConeTriangle(2))
        c4 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(0.0, [x[1:3]; x[5:6]])),
                [0.0]
            ), 
            MOI.Zeros(1)
        )

        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[7])], 0.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

        MatModel = MatOI.get_conic_form(Float64, model, [c1; c2; c3; c4])

        @test MatModel.c ≈ [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0]
        @test MatModel.b ≈ [0.0, 10.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0]
        A = MatModel.A
        @test A[2 , 1]  ≈  1.0
        @test A[3 , 1]  ≈  -1.0
        @test A[9 , 1]  ≈  -0.45
        @test A[10, 1]  ≈  0.45
        @test A[11, 1]  ≈  -0.45
        @test A[2 , 2]  ≈  1.0
        @test A[4 , 2]  ≈  -1.0
        @test A[9 , 2]  ≈  -0.8
        @test A[10, 2]  ≈  0.318198 atol=ATOL rtol=RTOL
        @test A[11, 2]  ≈  -0.1
        @test A[2 , 3]  ≈  1.0
        @test A[5 , 3]  ≈  -1.0
        @test A[9 , 3]  ≈  -0.9
        @test A[2 , 4]  ≈  1.0
        @test A[6 , 4]  ≈  -1.0
        @test A[9 , 4]  ≈  -0.225
        @test A[2 , 5]  ≈  1.0
        @test A[7 , 5]  ≈  -1.0
        @test A[9 , 5]  ≈  -0.1125
        @test A[10, 5]  ≈  0.1125
        @test A[11, 5]  ≈  -0.1125
        @test A[2 , 6]  ≈  1.0
        @test A[8 , 6]  ≈  -1.0
        @test A[11, 6]  ≈  -0.225
        @test A[9 , 7]  ≈  1.0
        @test A[11, 7]  ≈  1.0
    end
end

@testset "Testing restructure_arrays" begin
    s = [-5.8304524274385224e-18 1.672022895212547e-17 0.25440785071939953 0.17989351643078233 0.17989351643078233 0.21725123843399946 -0.36765368641915924 0.30723964791089897 0.31108967225522005 -0.3676536864191592 0.2172512384339987]
    y = [0.5447582105062065 0.3219045549528856 0.45524178943652027 -0.3219045563900619 -0.3219045563900619 1.133337233849821 0.9589717750123417 -0.4552417892860504 1.1333372326447106 0.958971775012335 1.1333372338498173]
    cones = MOI.AbstractVectorSet[MOI.PositiveSemidefiniteConeTriangle(3), MOI.SecondOrderCone(3), MOI.Zeros(1), MOI.Zeros(1)]

    _s, _y = MatOI.restructure_arrays(s,y,cones)

    @test _s ≈ [[-5.8304524274385224e-18, 1.672022895212547e-17, 0.25440785071939953, 0.17989351643078233, 0.17989351643078233, 0.21725123843399946], [-0.36765368641915924, 0.30723964791089897, 0.31108967225522005], [-0.3676536864191592], [0.2172512384339987]] atol=ATOL rtol=RTOL
    @test _y ≈ [[0.5447582105062065, 0.3219045549528856, 0.45524178943652027, -0.3219045563900619, -0.3219045563900619, 1.133337233849821], [0.9589717750123417, -0.4552417892860504, 1.1333372326447106], [0.958971775012335], [1.1333372338498173]] atol=ATOL rtol=RTOL
end


@testset "Testing minor utilities" begin
    # vector, matrix dimensions
    i = rand(10:100)
    @test MatOI.sympackedlen(MatOI.sympackeddim(i)) <= i
end
