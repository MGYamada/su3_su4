struct Block{Nc}
    length::Int64
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list_old::Vector{Int64}
    mβ_list::Vector{Int64}
    scalar_dict::Dict{Symbol, Vector{Matrix{ComplexF64}}}
    tensor_dict::Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}
end

struct EnlargedBlock{Nc}
    length::Int64
    α_list::Vector{SUNIrrep{Nc}}
    mα_list::Vector{Int64}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list::Vector{Int64}
    mαβ::Matrix{Int64}
    scalar_dict::Dict{Symbol, Vector{AbstractMatrix{ComplexF64}}}
    tensor_dict::Dict{Symbol, Matrix{Vector{AbstractMatrix{ComplexF64}}}}
end

"""
block_enl = enlarge_block(block, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = false)
enlarge the block by one site
"""
function enlarge_block(block::Block{Nc}, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = false) where Nc
    if rank == 0
        αs = copy(block.β_list)
        lenα = length(αs)
        funda = zeros(Int64, Nc)
        funda[1] = 1
        fundairrep = SUNIrrep(Tuple(funda))
        antiirrep = conj(fundairrep)
        adjoint = ones(Int64, Nc)
        adjoint[1] = 2
        adjoint[end] = 0
        adjointirrep = SUNIrrep(Tuple(adjoint))

        dest = map(α -> union([α], Set(keys(directproduct(α, fundairrep)))), αs)
        βs = filter!(β -> weight(β)[1] <= widthmax, collect(union(dest[block.mβ_list .> 0]...)))
        αβmatrix = [β ∈ d for d in dest, β in βs]
        mαβ = Diagonal(block.mβ_list) * αβmatrix
        @. αβmatrix = mαβ > 0
        lenβ = length(βs)
        cum_mαβ = vcat(zeros(Int64, 1, lenβ), cumsum(mαβ, dims = 1))
        ms = cum_mαβ[end, :]

        fac1 = sqrt((Nc ^ 2 - 1) / Nc)
        dp = [directproduct(βs[j], adjointirrep) for j in 1 : lenβ]
        Stask = map([(i, j) for i in 1 : lenβ, j in 1 : lenβ]) do (i, j)
            Threads.@spawn begin
                o1 = get(dp[j], βs[i], 0)
                map(1 : o1) do τ1
                    rtn = zeros(ComplexF64, ms[i], ms[j])
                    for k in 1 : lenα
                        if αs[k] != βs[i] && αs[k] != βs[j] && αβmatrix[k, i] && αβmatrix[k, j]
                            coeff = tables[2][αs[k], βs[j], βs[i]]
                            for l in 1 : mαβ[k, i]
                                rtn[cum_mαβ[k, i] + l, cum_mαβ[k, j] + l] = fac1 * coeff[τ1]
                            end
                        end
                    end
                    sparse(rtn)
                end
            end
        end
        Stemp = fetch.(Stask)
        MPI.bcast(ms, 0, comm)
        om = length.(Stemp)
        MPI.bcast(om, 0, comm)
    else
        ms = MPI.bcast(nothing, 0, comm)
        lenβ = length(ms)
        om = MPI.bcast(nothing, 0, comm)
        Stemp = [[spzeros(ComplexF64, ms[i], ms[j]) for τ1 in 1 : om[i, j]] for i in 1 : lenβ, j in 1 : lenβ]
    end

    for i in 1 : lenβ, j in 1 : lenβ
        for τ1 in 1 : om[i, j]
            if rank == 0
                MPI.bcast(Stemp[i, j][τ1], 0, comm)
            else
                Stemp[i, j][τ1] .= MPI.bcast(nothing, 0, comm)
            end
        end
    end

    if gpu
        Snew = [CUSPARSE.CuSparseMatrixCSC.(Stemp[i, j]) for i in 1 : lenβ, j in 1 : lenβ]
    else
        Snew = Stemp
    end

    if rank == 0
        fac2 = (Nc ^ 2 - 1) / sqrt(Nc) * signfactor
        # fac4 = 1.0
        Htask = map(1 : lenβ) do k
            Threads.@spawn begin
                rtn = !any(αβmatrix[:, k]) ? zeros(ComplexF64, 0, 0) : cat(block.scalar_dict[:H][αβmatrix[:, k]]..., dims = (1, 2))
                for i in 1 : lenα
                    if αβmatrix[i, k]
                        for j in 1 : lenα
                            if αβmatrix[j, k]
                                if αs[j] != βs[k] && αs[i] != βs[k]
                                    o1 = length(block.tensor_dict[:conn_S][i, j])
                                    if o1 > 0
                                        coeff = tables[3][αs[j], βs[k], αs[i]]
                                        for τ1 in 1 : o1
                                            @views axpy!(fac2 * coeff[τ1], block.tensor_dict[:conn_S][i, j][τ1], rtn[cum_mαβ[i, k] + 1 : cum_mαβ[i + 1, k], cum_mαβ[j, k] + 1 : cum_mαβ[j + 1, k]])
                                        end
                                    end
                                # elseif αs[i] == βs[k] && αs[j] != βs[k]
                                #     o1 = length(block.tensor_dict[:conn_U][i, j])
                                #     for τ1 in 1 : o1
                                #         @views axpy!(fac4 / Nc, block.tensor_dict[:conn_U][i, j][τ1], rtn[cum_mαβ[i, k] + 1 : cum_mαβ[i + 1, k], cum_mαβ[j, k] + 1 : cum_mαβ[j + 1, k]])
                                #     end
                                # elseif αs[j] == βs[k] && αs[i] != βs[k]
                                #     o1 = length(block.tensor_dict[:conn_D][i, j])
                                #     if o1 > 0
                                #         coeff = tables[8][βs[k], αs[i]]
                                #         for τ1 in 1 : o1
                                #             @views axpy!(fac4 * coeff, block.tensor_dict[:conn_D][i, j][τ1], rtn[cum_mαβ[i, k] + 1 : cum_mαβ[i + 1, k], cum_mαβ[j, k] + 1 : cum_mαβ[j + 1, k]])
                                #         end
                                #     end
                                else
                                    # o1 = length(block.tensor_dict[:conn_T][i, j])
                                    # for τ1 in 1 : o1
                                    #     @views axpy!(1.0, block.tensor_dict[:conn_T][i, j][τ1], rtn[cum_mαβ[i, k] + 1 : cum_mαβ[i + 1, k], cum_mαβ[j, k] + 1 : cum_mαβ[j + 1, k]])
                                    # end
                                end
                            end
                        end
                    end
                end
                rtn
            end
        end
        Htemp = fetch.(Htask)
    else
        Htemp = [Matrix{ComplexF64}(undef, m, m) for m in ms]
    end

    for k in 1 : lenβ
        MPI.Bcast!(Htemp[k], 0, comm)
    end

    if gpu
        Hnew = CuArray.(Htemp)
    else
        Hnew = Htemp
    end

    if rank == 0
        block_enl = EnlargedBlock(block.length + 1, αs, copy(block.mβ_list_old), βs, ms, mαβ, Dict{Symbol, Vector{AbstractMatrix{ComplexF64}}}(:H => Hnew), Dict{Symbol, Matrix{Vector{AbstractMatrix{ComplexF64}}}}(:conn_S => Snew))
    else
        block_enl = EnlargedBlock(block.length + 1, SUNIrrep{Nc}[], Int64[], SUNIrrep{Nc}[], Int64[], zeros(Int64, 0, 0), Dict{Symbol, Vector{AbstractMatrix{ComplexF64}}}(:H => Hnew), Dict{Symbol, Matrix{Vector{AbstractMatrix{ComplexF64}}}}(:conn_S => Snew))
    end

    block_enl
end

"""
newblock, newblock_enl, energy, Ψ0, trmat = dmrg_step(sys_enl, env_enl, m, widthmax, target, signfactor, comm, rank, Ncpu, tables; Ψ0_guess = nothing, gpu = false, Ngpu = 0)
a single step for DMRG
"""
function dmrg_step(sys_enl::EnlargedBlock{Nc}, env_enl::EnlargedBlock{Nc}, m, widthmax, target, signfactor, comm, rank, Ncpu, tables; Ψ0_guess = nothing, gpu = false, Ngpu = 0) where Nc
    sys_βs = MPI.bcast(sys_enl.β_list, 0, comm)
    env_βs = MPI.bcast(env_enl.β_list, 0, comm)
    sys_ms = MPI.bcast(sys_enl.mβ_list, 0, comm)
    env_ms = MPI.bcast(env_enl.mβ_list, 0, comm)

    γ = zeros(Int64, Nc)
    γirrep = SUNIrrep(Tuple(γ))
    T = OM_matrix(sys_βs, env_βs, γirrep) .> 0

    sys_len = length(sys_βs)
    env_len = length(env_βs)

    superblock_H1 = Tuple{AbstractMatrix{ComplexF64}, AbstractMatrix{ComplexF64}, Int64, Int64}[]
    superblock_H2 = Tuple{Vector{Tuple{ComplexF64, AbstractMatrix{ComplexF64}, AbstractMatrix{ComplexF64}, Int64, Int64}}, Int64, Int64, Int64, Int64}[]

    for k1 in 1 : sys_len, k2 in 1 : env_len
        if T[k1, k2] && (k1 + k2 - 2) % Ncpu == rank
            push!(superblock_H1, (sys_enl.scalar_dict[:H][k1], env_enl.scalar_dict[:H][k2], k1, k2))
        end
    end

    fac1 = sqrt(Nc ^ 2 - 1) * signfactor
    for k1 in 1 : sys_len, k2 in 1 : env_len
        if T[k1, k2]
            miniblock = Tuple{ComplexF64, AbstractMatrix{ComplexF64}, AbstractMatrix{ComplexF64}, Int64, Int64}[]
            for k3 in 1 : sys_len, k4 in 1 : env_len
                if T[k3, k4] && (k3 + k4 - 2) % Ncpu == rank
                    syssector = sys_enl.tensor_dict[:conn_S][k3, k1]
                    o1 = length(syssector)
                    envsector = env_enl.tensor_dict[:conn_S][k4, k2]
                    o2 = length(envsector)
                    if o1 > 0 && o2 > 0
                        cmatrix = tables[4][sys_βs[k1], env_βs[k2], γirrep, sys_βs[k3], env_βs[k4]]
                        for τ1 in 1 : o1, τ2 in 1 : o2
                            push!(miniblock, (fac1 * cmatrix[τ1, τ2], syssector[τ1], envsector[τ2], k3, k4))
                        end
                    end
                end
            end
            push!(superblock_H2, (miniblock, k1, k2, sys_ms[k1], env_ms[k2]))
        end
    end

    if isnothing(Ψ0_guess)
        if gpu
            initial = [[CUDA.rand(ComplexF64, env_ms[ki], sys_ms[kj]) for J in 1 : (T[kj, ki] && (kj + ki - 2) % Ncpu == rank)] for ki in 1 : env_len, kj in 1 : sys_len]
        else
            initial = [[rand(ComplexF64, env_ms[ki], sys_ms[kj]) for J in 1 : (T[kj, ki] && (kj + ki - 2) % Ncpu == rank)] for ki in 1 : env_len, kj in 1 : sys_len]
        end
    else
        initial = Ψ0_guess
    end

    @time E, Ψ0 = Lanczos!(initial, target + 1, comm, rank) do Ψout, Ψin
        for (left, right, leftind, rightind) in superblock_H1
            temp1 = Ψin[rightind, leftind][1] * transpose(left)
            temp2 = right * Ψin[rightind, leftind][1]
            @. Ψout[rightind, leftind][1] = temp1 + temp2
        end

        for (miniblock, leftin, rightin, leftsize, rightsize) in superblock_H2
            root = (leftin + rightin - 2) % Ncpu
            if rank == root
                temp3 = Ψin[rightin, leftin][1]
            else
                if gpu
                    temp3 = CuMatrix{ComplexF64}(undef, rightsize, leftsize)
                else
                    temp3 = Matrix{ComplexF64}(undef, rightsize, leftsize)
                end
            end
            MPI.Bcast!(temp3, root, comm)
            for (coeff, left, right, leftout, rightout) in miniblock
                temp4 = transpose(left * transpose(right * temp3))
                @. Ψout[rightout, leftout][1] += coeff * temp4
            end
        end


    end

    balancer = zeros(Int64, sys_len)
    if rank == 0
        if Ncpu > 1
            max_m = maximum(sys_ms)
            sys_loads = @. (sys_ms / max_m) ^ 3
            load_sum = zeros(Ncpu)
            load_sum[1] = sum(sys_loads)
            load_max = maximum(load_sum)
            for i in 1 : 100
                j = rand(1 : sys_len)
                new_b = rand(filter(x -> x != balancer[j], 0 : Ncpu - 1))
                load_sum[balancer[j] + 1] -= sys_loads[j]
                load_sum[new_b + 1] += sys_loads[j]
                new_load_max = maximum(load_sum)
                if log(rand()) < 100.0 * (load_max - new_load_max)
                    balancer[j] = new_b
                    load_max = new_load_max
                else
                    load_sum[balancer[j] + 1] += sys_loads[j]
                    load_sum[new_b + 1] -= sys_loads[j]
                end
            end
        end
    end
    MPI.Bcast!(balancer, 0, comm)

    energy = E #+ (sys_enl.length + env_enl.length - 1) / Nc
    dimβ = dim.(sys_βs)
    ρs = Matrix{ComplexF64}[]

    for k in 1 : sys_len
        fac = 1.0 / dimβ[k]
        if gpu
            ρgpu = CUDA.zeros(ComplexF64, sys_ms[k], sys_ms[k])
            for j in 1 : env_len
                for J in eachindex(Ψ0[j, k])
                    CUBLAS.herk!('U', 'C', fac, Ψ0[j, k][J], 1.0, ρgpu)
                end
            end
            ρ = Array(ρgpu)
            MPI.Reduce!(ρ, +, balancer[k], comm)
            if rank == balancer[k]
                push!(ρs, ρ)
            end
        else
            ρ = zeros(ComplexF64, sys_ms[k], sys_ms[k])
            for j in 1 : env_len
                for J in eachindex(Ψ0[j, k])
                    BLAS.herk!('U', 'C', fac, Ψ0[j, k][J], 1.0, ρ)
                end
            end
            MPI.Reduce!(ρ, +, balancer[k], comm)
            if rank == balancer[k]
                push!(ρs, ρ)
            end
        end
    end

    @time if gpu
        λζtemp = map(ρ -> magma_syevd!('V', 'U', ρ), ρs)
    else
        λζtemp = map(ρ -> LAPACK.syev!('V', 'U', ρ), ρs)
    end

    MPI.Barrier(comm)

    if rank == 0
        λ = [Vector{Float64}(undef, sys_ms[k]) for k in 1 : sys_len]
        ζ = [Matrix{ComplexF64}(undef, sys_ms[k], sys_ms[k]) for k in 1 : sys_len]
        @. λ[balancer .== 0] = first(λζtemp)
        @. ζ[balancer .== 0] = last(λζtemp)
    end
    for k in 1 : sys_len
        if balancer[k] != 0
            if rank == balancer[k]
                pos = count(balancer[1 : k] .== rank)
                MPI.Send(λζtemp[pos][1], 0, balancer[k], comm)
                MPI.Send(λζtemp[pos][2], 0, balancer[k] + Ncpu, comm)
            elseif rank == 0
                MPI.Recv!(λ[k], balancer[k], balancer[k], comm)
                MPI.Recv!(ζ[k], balancer[k], balancer[k] + Ncpu, comm)
            end
        end
    end

    MPI.Barrier(comm)

    if rank == 0
        Λ = sort!(vcat(reverse.(λ)...), rev = true)
        λthreshold = m < length(Λ) ? Λ[m + 1] : -Inf
        indices = map(x -> x .> λthreshold, λ)
        λnew = map(x -> x[1][x[2]], zip(λ, indices))
        if gpu
            transformation_matrix = map(x -> CuArray(x[1][:, x[2]]), zip(ζ, indices))
        else
            transformation_matrix = map(x -> x[1][:, x[2]], zip(ζ, indices))
        end

        msnew = map(k -> size(transformation_matrix[k], 2), 1 : sys_len)
        println("Keeping ", sum(msnew), " SU($Nc) states corresponding to ", sum(dimβ .* msnew), " U(1) states")
        Hnew = map(k -> size(transformation_matrix[k], 2) == 0 ? zeros(ComplexF64, 0, 0) : Array(transpose(transformation_matrix[k]) * (sys_enl.scalar_dict[:H][k] * conj.(transformation_matrix[k]))), 1 : sys_len)
        Snew = map(k -> [size(transformation_matrix[k[2]], 2) == 0 ? zeros(ComplexF64, size(transformation_matrix[k[1]], 2), 0) : Array(transpose(transformation_matrix[k[1]]) * (M * conj.(transformation_matrix[k[2]]))) for M in sys_enl.tensor_dict[:conn_S][k...]], [(ki, kj) for ki in 1 : sys_len, kj in 1 : sys_len])
        newblock = Block(sys_enl.length, sys_enl.β_list, sys_enl.mβ_list, msnew, Dict{Symbol, Vector{Matrix{ComplexF64}}}(:H => Hnew), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}(:conn_S => Snew))
    else
        newblock = Block(sys_enl.length, SUNIrrep{Nc}[], Int64[], Int64[], Dict{Symbol, Vector{Matrix{ComplexF64}}}(), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}())
        transformation_matrix = nothing
    end

    newblock_enl = enlarge_block(newblock, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = gpu)

    if rank == 0
        truncation_error = 1.0 - sum(sum(vcat(λnew[newblock_enl.mαβ[:, l] .> 0]...); init = 0.0) * dim(β) for (l, β) in enumerate(newblock_enl.β_list)) / (Nc + 1)
        println("truncation error: ", truncation_error)
    end

    if !isnothing(Ψ0_guess)
        nume = MPI.Reduce(real(dot(Ψ0_guess, Ψ0)), +, 0, comm)
        den1 = MPI.Reduce(real(dot(Ψ0_guess, Ψ0_guess)), +, 0, comm)
        den2 = MPI.Reduce(real(dot(Ψ0, Ψ0)), +, 0, comm)
        if rank == 0
            println("overlap |<ψ_guess|ψ>| = ", abs(nume) / sqrt(den1 * den2))
        end
    end

    newblock, newblock_enl, energy, Ψ0, transformation_matrix
end

"""
string = graphic(sys_block, env_block; sys_label = :l)
visualizes DMRG
"""
function graphic(sys_block, env_block; sys_label = :l)
    str = repeat("=", sys_block.length) * "**" * repeat("-", env_block.length)
    if sys_label == :r
        str = reverse(str)
    elseif sys_label != :l
        throw(ArgumentError("sys_label must be :l or :r"))
    end
    str
end

"""
finite_system_algorithm(Nc, L, m_warmup, m_sweep_list, widthmax, target, tables; gpu = false, fileio = false, scratch = ".")
doing the finite-system algorithm
(target = 0: ground state, target = 1: 1st excited state...)
Currently only suupports a small number for target
"""
function finite_system_algorithm(Nc, L, m_warmup, m_sweep_list, widthmax, target, tables; gpu = false, fileio = false, scratch = ".")
    @assert iseven(L)
    @assert L % Nc == 0

    MPI.Init_thread(MPI.THREAD_FUNNELED)
    γ = zeros(Int64, Nc)
    γirrep = SUNIrrep(Tuple(γ))

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Ncpu = MPI.Comm_size(comm)

    if gpu
        Ngpu = Int(length(devices()))
        @assert Ncpu <= Ngpu
        device!(rank)
        magma_init()
    else
        Ngpu = 0
    end

    signfactor = iseven(Nc) ? -1.0 : 1.0
    trivialirrep = SUNIrrep(Tuple(zeros(Int64, Nc)))

    if rank == 0
        println(repeat("-", 60))
        println("SU($Nc) DMRG simulation with a length $L:")
        irreps = irreplist(Nc, widthmax)
        println(length(irreps), " irreps from ", weight(first(irreps)), " to ", weight(last(irreps)), " are used in the calculation.")
        println(target == 0 ? "The ground state" : "The excited state #$target", " will be calculated.")
        println(repeat("-", 60))

        if fileio
            dirid = lpad(rand(0 : 99999), 5, "0")
            mkdir("$scratch/temp$dirid")
        else
            block_table = Dict{Tuple{Symbol, Int64}, Block{Nc}}()
            trmat_table = Dict{Tuple{Symbol, Int64}, Vector{Matrix{ComplexF64}}}()
        end

        funda = zeros(Int64, Nc)
        funda[1] = 1
        fundairrep = SUNIrrep(Tuple(funda))
        β1 = [fundairrep, trivialirrep]
        H1 = [zeros(ComplexF64, 1, 1), zeros(ComplexF64, 1, 1)]
        S1 = [Matrix{ComplexF64}[] for i in 1 : 2, j in 1 : 2]
        S1[1, 1] = [ones(ComplexF64, 1, 1) .* sqrt((Nc ^ 2 - 1) / Nc)]
        block = Block(1, β1, [1, 1], [1, 1], Dict{Symbol, Vector{Matrix{ComplexF64}}}(:H => H1), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}(:conn_S => S1))
        trmat = [diagm([one(ComplexF64)]), diagm([one(ComplexF64)])]

        if fileio
            jldsave("$scratch/temp$dirid/block_l_$(block.length).jld2"; env_block = block)
            jldsave("$scratch/temp$dirid/block_r_$(block.length).jld2"; env_block = block)
            jldsave("$scratch/temp$dirid/trmat_l_$(block.length).jld2"; env_trmat = Array.(trmat))
            jldsave("$scratch/temp$dirid/trmat_r_$(block.length).jld2"; env_trmat = Array.(trmat))
        else
            block_table[:l, block.length] = block
            block_table[:r, block.length] = block
            trmat_table[:l, block.length] = Array.(trmat)
            trmat_table[:r, block.length] = Array.(trmat)
        end

        println("#")
        println("# Infinite-system algorithm with m = ", m_warmup)
        println("#")
    else
        block = Block(1, SUNIrrep{Nc}[], Int64[], Int64[], Dict{Symbol, Vector{Matrix{ComplexF64}}}(), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}())
    end

    block_enl = enlarge_block(block, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = gpu) # 合体
    Ψ0 = nothing

    while 2block.length < L
        if rank == 0
            println(graphic(block, block))
        end

        if 2block.length < L - 2
            block, block_enl, energy, Ψ0, trmat = dmrg_step(block_enl, block_enl, m_warmup, widthmax, target, signfactor, comm, rank, Ncpu, tables; gpu = gpu, Ngpu = Ngpu)
        else
            block, block_enl, energy, Ψ0, trmat = dmrg_step(block_enl, block_enl, m_warmup, widthmax, target, signfactor, comm, rank, Ncpu, tables; gpu = gpu, Ngpu = Ngpu)
        end

        if rank == 0
            println("E / L = ", energy / 2block.length)
            println("E     = ", energy)
            if fileio
                jldsave("$scratch/temp$dirid/block_l_$(block.length).jld2"; env_block = block)
                jldsave("$scratch/temp$dirid/block_r_$(block.length).jld2"; env_block = block)
                jldsave("$scratch/temp$dirid/trmat_l_$(block.length).jld2"; env_trmat = Array.(trmat))
                jldsave("$scratch/temp$dirid/trmat_r_$(block.length).jld2"; env_trmat = Array.(trmat))
            else
                block_table[:l, block.length] = block
                block_table[:r, block.length] = block
                trmat_table[:l, block.length] = Array.(trmat)
                trmat_table[:r, block.length] = Array.(trmat)
            end
        end
    end

    sys_label, env_label = :l, :r
    sys_block = block
    sys_trmat = trmat
    sys_block_enl = block_enl

    if rank == 0
        if fileio
            env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_block.length - 1).jld2")
            if gpu
                env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 1).jld2"))
            else
                env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 1).jld2")
            end
        else
            env_block = block_table[env_label, L - sys_block.length - 1]
            if gpu
                env_trmat = CuArray.(trmat_table[env_label, L - sys_block.length - 1])
            else
                env_trmat = trmat_table[env_label, L - sys_block.length - 1]
            end
        end
    else
        env_block = Block(L - sys_block.length - 1, SUNIrrep{Nc}[], Int64[], Int64[], Dict{Symbol, Vector{Matrix{ComplexF64}}}(), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}())
        env_trmat = Matrix{ComplexF64}[]
    end

    env_block_enl = enlarge_block(env_block, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = gpu)

    for m in m_sweep_list
        if rank == 0
            println("#")
            println("# Performing sweep with m = ", m)
            println("#")
        end
        while true
            if rank == 0
                sys_mαβ = MPI.bcast(sys_block_enl.mαβ, 0, comm)
                env_mαβ = MPI.bcast(env_block_enl.mαβ, 0, comm)
            else
                sys_mαβ = MPI.bcast(nothing, 0, comm)
                env_mαβ = MPI.bcast(nothing, 0, comm)
            end
            cum_sys_mαβ = vcat(zeros(Int64, 1, size(sys_mαβ, 2)), cumsum(sys_mαβ, dims = 1))
            cum_env_mαβ = vcat(zeros(Int64, 1, size(env_mαβ, 2)), cumsum(env_mαβ, dims = 1))
            if rank == 0
                sys_αs = MPI.bcast(sys_block_enl.α_list, 0, comm)
                sys_βs = MPI.bcast(sys_block_enl.β_list, 0, comm)
                env_αs = MPI.bcast(env_block_enl.α_list, 0, comm)
                env_βs = MPI.bcast(env_block_enl.β_list, 0, comm)
            else
                sys_αs = MPI.bcast(nothing, 0, comm)
                sys_βs = MPI.bcast(nothing, 0, comm)
                env_αs = MPI.bcast(nothing, 0, comm)
                env_βs = MPI.bcast(nothing, 0, comm)
            end
            T = OM_matrix(sys_βs, env_αs, trivialirrep) .> 0
            if rank == 0
                sys_mα = MPI.bcast(sys_block_enl.mα_list, 0, comm)
                sys_mβ = MPI.bcast(sys_block_enl.mβ_list, 0, comm)
                env_mα = MPI.bcast(env_block_enl.mα_list, 0, comm)
                env_mβ = MPI.bcast(env_block_enl.mβ_list, 0, comm)
            else
                sys_mα = MPI.bcast(nothing, 0, comm)
                sys_mβ = MPI.bcast(nothing, 0, comm)
                env_mα = MPI.bcast(nothing, 0, comm)
                env_mβ = MPI.bcast(nothing, 0, comm)
            end
            # if gpu
            #     Ψ0_guess = [[CUDA.zeros(ComplexF64, mi, mj) for J in 1 : (T[j, i] && (j + i - 2) % Ncpu == rank)] for (i, mi) in enumerate(env_mα), (j, mj) in enumerate(sys_mβ)]
            # else
            #     Ψ0_guess = [[zeros(ComplexF64, mi, mj) for J in 1 : (T[j, i] && (j + i - 2) % Ncpu == rank)] for (i, mi) in enumerate(env_mα), (j, mj) in enumerate(sys_mβ)]
            # end
            # if rank == 0
            #     for j in 1 : length(sys_αs)
            #         MPI.bcast(size(sys_trmat[j]), 0, comm)
            #         MPI.Bcast!(sys_trmat[j], 0, comm)
            #     end
            #     for i in 1 : length(env_αs)
            #         MPI.bcast(size(env_trmat[i]), 0, comm)
            #         MPI.Bcast!(env_trmat[i], 0, comm)
            #     end
            # else
            #     if gpu
            #         sys_trmat = CuMatrix{ComplexF64}[]
            #     else
            #         sys_trmat = Matrix{ComplexF64}[]
            #     end
            #     for j in 1 : length(sys_αs)
            #         x, y = MPI.bcast(nothing, 0, comm)
            #         if gpu
            #             push!(sys_trmat, CuMatrix{ComplexF64}(undef, x, y))
            #         else
            #             push!(sys_trmat, Matrix{ComplexF64}(undef, x, y))
            #         end
            #         MPI.Bcast!(sys_trmat[j], 0, comm)
            #     end
            #     if gpu
            #         env_trmat = CuMatrix{ComplexF64}[]
            #     else
            #         env_trmat = Matrix{ComplexF64}[]
            #     end
            #     for i in 1 : length(env_αs)
            #         x, y = MPI.bcast(nothing, 0, comm)
            #         if gpu
            #             push!(env_trmat, CuMatrix{ComplexF64}(undef, x, y))
            #         else
            #             push!(env_trmat, Matrix{ComplexF64}(undef, x, y))
            #         end
            #         MPI.Bcast!(env_trmat[i], 0, comm)
            #     end
            # end
            # for (k, βk) in enumerate(env_βs)
            #     αj = conj(βk)
            #     j = findfirst(isequal(αj), sys_αs)
            #     if !isnothing(j)
            #         root = (k + j - 2) % Ncpu
            #         if rank == root
            #             temp1 = Ψ0[k, j][1]
            #         else
            #             if gpu
            #                 temp1 = CuMatrix{ComplexF64}(undef, env_mβ[k], sys_mα[j])
            #             else
            #                 temp1 = Matrix{ComplexF64}(undef, env_mβ[k], sys_mα[j])
            #             end
            #         end
            #         MPI.Bcast!(temp1, root, comm)
            #         temp2 = temp1 * sys_trmat[j]
            #         for i in findall(x -> x > 0, env_mαβ[:, k])
            #             αi = env_αs[i]
            #             βl = conj(αi)
            #             l = findfirst(isequal(βl), sys_βs)
            #             if !isnothing(l) && (i + l - 2) % Ncpu == rank && sys_mαβ[j, l] > 0
            #                 temp3 = env_trmat[i] * temp2[cum_env_mαβ[i, k] + 1 : cum_env_mαβ[i + 1, k], :]
            #                 @. Ψ0_guess[i, l][1][:, cum_sys_mαβ[j, l] + 1 : cum_sys_mαβ[j + 1, l]] += tables[5][αj, βl, αi, γirrep, βk] * temp3 # fix later
            #             end
            #         end
            #     end
            # end

            if rank == 0
                if fileio
                    env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_block.length - 2).jld2")
                    if gpu
                        env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 2).jld2"))
                    else
                        env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 2).jld2")
                    end
                else
                    env_block = block_table[env_label, L - sys_block.length - 2]
                    if gpu
                        env_trmat = CuArray.(trmat_table[env_label, L - sys_block.length - 2])
                    else
                        env_trmat = trmat_table[env_label, L - sys_block.length - 2]
                    end
                end
            else
                env_block = Block(L - sys_block.length - 2, SUNIrrep{Nc}[], Int64[], Int64[], Dict{Symbol, Vector{Matrix{ComplexF64}}}(), Dict{Symbol, Matrix{Vector{Matrix{ComplexF64}}}}())
                env_trmat = Matrix{ComplexF64}[]
            end

            env_block_enl = enlarge_block(env_block, widthmax, signfactor, comm, rank, Ncpu, tables; gpu = gpu)

            if env_block.length == 1
                sys_block, env_block = env_block, sys_block
                sys_trmat, env_trmat = env_trmat, sys_trmat
                sys_block_enl, env_block_enl = env_block_enl, sys_block_enl
                sys_label, env_label = env_label, sys_label
                # Ψ0_guess = permutedims(map(x -> permutedims.(x), Ψ0_guess))
            end

            if rank == 0
                println(graphic(sys_block, env_block; sys_label = sys_label))
            end

            sys_block, sys_block_enl, energy, Ψ0, sys_trmat = dmrg_step(sys_block_enl, env_block_enl, m, widthmax, target, signfactor, comm, rank, Ncpu, tables; gpu = gpu, Ngpu = Ngpu)

            if rank == 0
                if fileio
                    jldsave("$scratch/temp$dirid/block_$(sys_label)_$(sys_block.length).jld2"; env_block = sys_block)
                    jldsave("$scratch/temp$dirid/trmat_$(sys_label)_$(sys_block.length).jld2"; env_trmat = Array.(sys_trmat))
                else
                    block_table[sys_label, sys_block.length] = sys_block
                    trmat_table[sys_label, sys_block.length] = Array.(sys_trmat)
                end

                println("E / L = ", energy / L)
                println("E     = ", energy)
            end

            if sys_label == :l && 2sys_block.length == L
                break
            end
        end
    end

    if rank == 0 && fileio
        rm("$scratch/temp$dirid"; recursive = true)
    end

    if gpu
        magma_finalize()
    end

    MPI.Finalize()
end

"""
infinite_system_algorithm(Nc, L, m, widthmax, target, tables; gpu = false, fileio = false, scratch = ".")
doing the infinite-system algorithm
(target = 0: ground state, target = 1: 1st excited state...)
Currently only suupports a small number for target
"""
function infinite_system_algorithm(Nc, L, m, widthmax, target, tables; gpu = false, fileio = false, scratch = ".")
    finite_system_algorithm(Nc, L, m, Int64[], widthmax, target, tables; gpu = gpu, fileio = fileio, scratch = scratch)
end