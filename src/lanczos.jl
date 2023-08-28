const tol_wavefunction = 1e-13
const tol_Lanczos = 1e-13

MyMatrix = Union{Matrix{Vector{Matrix{ComplexF64}}}, Matrix{Vector{CuMatrix{ComplexF64}}}}

function LinearAlgebra.dot(x::MyMatrix, y::MyMatrix)
    s = 0.0im
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            s += dot(x[IX][JX], y[IY][JY])
        end
    end
    s
end

function LinearAlgebra.axpy!(α, x::MyMatrix, y::MyMatrix)
    for (IY, IX) in zip(eachindex(y), eachindex(x))
        for (JY, JX) in zip(eachindex(y[IY]), eachindex(x[IX]))
            @. y[IY][JY] += α * x[IX][JX]
        end
    end
    y
end

function LinearAlgebra.axpby!(α, x::MyMatrix, β, y::MyMatrix)
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            @. y[IY][JY] = α * x[IX][JX] + β * y[IY][JY]
        end
    end
    y
end

function LinearAlgebra.rmul!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] .*= b
        end
    end
    A
end

function LinearAlgebra.rdiv!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] ./= b
        end
    end
    A
end

function Base.copyto!(dest::MyMatrix, src::MyMatrix)
    for (IX, IY) in zip(eachindex(dest), eachindex(src))
        for (JX, JY) in zip(eachindex(dest[IX]), eachindex(src[IY]))
            dest[IX][JX] .= src[IY][JY]
        end
    end
    dest
end

"""
CG!(A!, val, x, Ax, buffer1, buffer2, comm, rank)
CG routine for the Lanczos method
"""
function CG!(A!::Function, val, x, Ax, buffer1, buffer2, comm, rank)
    valnew = 0.0
    r = Ax
    p = buffer1
    Ap = buffer2
    for i in 1 : 100
        valshift = val - 1e-8
        axpby!(1.0 + valshift, x, -1.0, r)
        copyto!(p, r)
        normold = MPI.Allreduce(real(dot(r, r)), +, comm)
        j = 1
        while true
            A!(Ap, p)
            axpy!(-valshift, p, Ap)
            α = normold / MPI.Allreduce(real(dot(p, Ap)), +, comm)
            axpy!(α, p, x)
            axpy!(-α, Ap, r)
            normnew = MPI.Allreduce(real(dot(r, r)), +, comm)
            if j == 10 || normnew < 1e-8
                break
            end
            β = normnew / normold
            axpby!(1.0, r, β, p)
            normold = normnew
            j += 1
        end
        rdiv!(x, sqrt(MPI.Allreduce(real(dot(x, x)), +, comm)))
        A!(r, x)
        valnew = MPI.Allreduce(real(dot(x, r)), +, comm)
        if abs((valnew - val) / valnew) < tol_wavefunction
            break
        end
        val = valnew
    end
    valnew
end

"""
Lanczos!(A!, initial, position, comm, rank; maxiter = 100)
returns eigenpairs of a linear map A!
"""
function Lanczos!(A!::Function, initial, position, comm, rank; maxiter = 100)
    ketkm1 = deepcopy(initial)
    for I in eachindex(ketkm1)
        for J in eachindex(ketkm1[I])
            ketkm1[I][J] .= 0.0
        end
    end
    rdiv!(initial, sqrt(MPI.Allreduce(real(dot(initial, initial)), +, comm)))
    ketk = deepcopy(initial)
    ketk1 = deepcopy(ketk)
    β = 0.0
    αlist = Float64[]
    βlist = Float64[]
    vals = Float64[]
    vecs = zeros(0, 0)
    vold = Inf
    k = 1
    while true
        A!(ketk1, ketk)
        α = MPI.Allreduce(real(dot(ketk, ketk1)), +, comm)
        push!(αlist, α)
        if k >= position
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            vals = MPI.bcast(vals, 0, comm)
            if k == maxiter || abs((vals[position] - vold) / vals[position]) < tol_Lanczos
                break
            end
            vold = vals[position]
        end
        axpy!(-β, ketkm1, ketk1)
        axpy!(-α, ketk, ketk1)
        β = sqrt(MPI.Allreduce(real(dot(ketk1, ketk1)), +, comm))
        if β == 0.0
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            vals = MPI.bcast(vals, 0, comm)
            break
        end
        rdiv!(ketk1, β)
        copyto!(ketkm1, ketk)
        copyto!(ketk, ketk1)
        push!(βlist, β)
        k += 1
    end
    vecs = MPI.bcast(vecs, 0, comm)

    for I in eachindex(ketkm1)
        for J in eachindex(ketkm1[I])
            ketkm1[I][J] .= 0.0
        end
    end
    copyto!(ketk, initial)
    β = 0.0
    position = min(position, size(vecs, 2))
    rmul!(initial, vecs[1, position])
    for k in 1 : size(vecs, 1) - 1
        A!(ketk1, ketk)
        α = αlist[k]
        axpy!(-β, ketkm1, ketk1)
        axpy!(-α, ketk, ketk1)
        β = βlist[k]
        rdiv!(ketk1, β)
        copyto!(ketkm1, ketk)
        copyto!(ketk, ketk1)
        axpy!(vecs[k + 1, position], ketk, initial)
    end
    rdiv!(initial, sqrt(MPI.Allreduce(real(dot(initial, initial)), +, comm)))
    A!(ketk, initial)
    val = MPI.Allreduce(real(dot(initial, ketk)), +, comm)
    val2 = val ^ 2
    var = MPI.Allreduce(real(dot(ketk, ketk)), +, comm)
    if abs((val - vals[position]) / val) < tol_wavefunction && abs((var - val2) / val2) < tol_wavefunction
        rtnval = val
    else
        rtnval = CG!(A!, val, initial, ketk, ketkm1, ketk1, comm, rank)
    end
    rtnval, initial
end