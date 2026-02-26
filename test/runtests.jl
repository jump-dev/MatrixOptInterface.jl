# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using SparseArrays, Test

import MatrixOptInterface

const MatOI = MatrixOptInterface
const MOI = MatOI.MOI
const MOIU = MatOI.MOIU
const MOIB = MOI.Bridges
const MOIT = MOI.Test

const ATOL = 1e-4
const RTOL = 1e-4

include("linear.jl")
include("conic_form.jl")
