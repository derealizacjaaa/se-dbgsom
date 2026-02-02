include(joinpath(@__DIR__, "..", "src", "DBGSOM.jl"))
using .BasicDBGSOM
using Test
using Statistics
using Random

include("test_nan_handling.jl")
include("test_qe_te.jl")
