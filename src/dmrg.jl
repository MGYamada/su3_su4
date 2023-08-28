using MPI

using LinearAlgebra
using SparseArrays
using SUNRepresentations
using JLD2
using CUDA

include("suncalc.jl")
include("lanczos.jl")
include("finite.jl")

function main()
    Nc = 3
    widthmax = 13
    @load "table_SU$(Nc)_$widthmax.jld2" tables
    finite_system_algorithm(Nc, 6, 100, [100, 100, 200, 200, 300, 300, 400, 400, 500, 500], widthmax, 0, tables; fileio = true)
    println("finished!")
end
main()