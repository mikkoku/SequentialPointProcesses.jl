using .CUDA

cuda_available = true

function compute_integral(m, qx, qy, xs, nx::NamedTuple{(:nx, :type, :batchsize)})
    integrate_cuda(m, nx.type, qx, qy, xs, nx.batchsize)
end

function integrate_cuda_inner!(m::Softcore, A::AbstractArray{T}, x, y, xy::AbstractVector{<:Tuple}, stride) where T
    dist2((x1, x2), (y1, y2)) = (x1-y1)^2 + (x2-y2)^2
    k0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if k0 <= stride[1] && j0 <= stride[2]
        for k in k0:stride[1]:length(x)
            for j = j0:stride[2]:length(y)
                z = zero(T)
                for l in 1:length(xy)
                    d = dist2((x[k],y[j]),xy[l]) :: T
                    d = sqrt(d) :: T
                    z += m.kernel_integral(d)
                    A[k0, j0, l] += exp(-z)
                end
            end
        end
    end
    nothing
end
function integrate_cuda(m, T::Type, x,y,xy, batchsize)
    A = CUDA.zeros(T, (cld(length(x), batchsize), cld(length(y), batchsize), length(xy)))
    x = convert(CuVector{T}, collect(x))
    y = convert(CuVector{T}, collect(y))
    xy = convert(CuVector{Tuple{T,T}}, xy)
    num_threads = (256,1)
    num_blocks = cld.(size(A)[1:2], num_threads)
    stride = size(A)[1:2]
    # @cuda blocks=num_blocks threads=num_threads integrate_cuda_inner!(m, A, x, y, xy, stride)
    kernel = @cuda launch=false integrate_cuda_inner!(
        m, A, x, y, xy, stride)
    config = launch_configuration(kernel.fun)
    # @info """memory: $(Base.format_bytes(sizeof(A)))
    #         $(CUDA.registers(kernel)) registers,  $(CUDA.maxthreads(kernel)) threads,
    #     - $(Base.format_bytes(CUDA.memory(kernel).local)) local memory,
    #         $(Base.format_bytes(CUDA.memory(kernel).shared)) shared memory,
    #         $(Base.format_bytes(CUDA.memory(kernel).constant)) constant memory
    #         $config, $((threads=num_threads, blocks=num_blocks))"""
    kernel(m, A, x, y, xy, stride;
        threads=num_threads, blocks=num_blocks)
    z = sum(A, dims=(1,2))
    a = collect(z)
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(z)
    a
end

function integrate_cuda_inner!(m::Softcore2, A::AbstractArray{T}, qx, qy,
    xy::AbstractVector{<:Tuple}, stride) where {T}
    d((x1, x2), (y1, y2)) = sqrt((x1-y1)^2 + (x2-y2)^2)
    k0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    @inbounds if k0 <= stride[1] && j0 <= stride[2]
        for k in k0:stride[1]:length(qx)
            for j = j0:stride[2]:length(qy)
                x = (qx[k], qy[j])
                # z = zero(T)
                for l in 1:length(xy)
                    R = m.R(l+1)
                    y = xy[1]
                    S = m.phi(d(x,y), R)
                    for i in 2:l
                        y = xy[i]
                        S = m.op(S, m.phi(d(x,y), R))
                    end

                    theta = m.theta(l+1)
                    psiS = m.psi(S, R)
                    A[k0, j0, l] += theta * psiS + (one(theta)-theta) * (one(psiS)-psiS)
                end
            end
        end
    end
    nothing
end

function integrate_cuda_inner!(m::Softcore2, A::AbstractArray{T}, qx, qy,
    xy_::AbstractVector{<:Tuple}, Rs_, thetas_, stride) where {T}
    # No difference between global, const or shared mem
    # Rs = CUDA.Const(Rs_)
    # thetas = CUDA.Const(thetas_)
    # xy = CUDA.Const(xy_)
    n = length(xy_)
    xy = @cuStaticSharedMem eltype(xy_) (256,)
    thetas = @cuStaticSharedMem eltype(thetas_) (256,)
    Rs = @cuStaticSharedMem eltype(Rs_) (256,)
    i = threadIdx().x + (threadIdx().y-1)*blockDim().x
    if i <= n
        xy[i] = xy_[i]
        thetas[i+1] = thetas_[i+1]
        Rs[i+1] = Rs_[i+1]
    end
    sync_threads()

    d((x1, x2), (y1, y2)) = sqrt((x1-y1)^2 + (x2-y2)^2)
    k0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    @inbounds    if k0 <= stride[1] && j0 <= stride[2]
        for k in k0:stride[1]:length(qx)
            for j = j0:stride[2]:length(qy)
                x = (qx[k], qy[j])
                for l in 1:n
                    R = Rs[l+1]
                    y = xy[1]
                    S = m.phi(d(x,y), R)
                    for i in 2:l
                        y = xy[i]
                        S = m.op(S, m.phi(d(x,y), R))
                    end

                    theta = thetas[l+1]
                    psiS = m.psi(S, R)
                    A[k0, j0, l] += theta * psiS + (one(theta)-theta) * (one(psiS)-psiS)
                end
            end
        end
    end
    nothing
end

function integrate_cuda(m::Softcore2v, T::Type, x,y,xy, batchsize)
    A = CUDA.zeros(T, (cld(length(x), batchsize), cld(length(y), batchsize), length(xy)))
    x = convert(CuVector{T}, collect(x))
    y = convert(CuVector{T}, collect(y))
    xy = convert(CuVector{Tuple{T,T}}, xy)
    R = convert(CuVector{T}, m.R)
    theta = convert(CuVector{T}, m.theta)
    num_threads = (30,29)
    num_blocks = cld.(size(A)[1:2], num_threads)
    stride = size(A)[1:2]
    m2 = Softcore2(k -> 1, m.psi, k -> 1, m.phi, m.op)
    # @cuda blocks=num_blocks threads=num_threads

    kernel = @cuda launch=false integrate_cuda_inner!(
        m2, A, x, y, xy, R, theta, stride)
    config = launch_configuration(kernel.fun)
    # @info """memory: $(Base.format_bytes(sizeof(A)))
    #         $(CUDA.registers(kernel)) registers,  $(CUDA.maxthreads(kernel)) threads,
    #     - $(Base.format_bytes(CUDA.memory(kernel).local)) local memory,
    #         $(Base.format_bytes(CUDA.memory(kernel).shared)) shared memory,
    #         $(Base.format_bytes(CUDA.memory(kernel).constant)) constant memory
    #         $config, $((threads=num_threads, blocks=num_blocks))"""
    kernel(m2, A, x, y, xy, R, theta, stride;
        threads=num_threads, blocks=num_blocks)
    z = sum(A, dims=(1,2))
    a = collect(z)
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(z)
    a
end
