using .CUDA

function compute_integral(qx, qy, xs, f, nx::NamedTuple{(:nx, type:, :batchsize)}) where f
    integrate_cuda(nx.type, qx, qy, xs, f, nx.batchsize)
end

function integrate_cuda_inner!(A::AbstractArray{T}, x, y, xy::AbstractVector{<:Tuple}, kernel::F, stride) where {T, F}
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
                    z += kernel(d)
                    A[k0, j0, l] += exp(-z)
                end
            end
        end
    end
    nothing
end
function integrate_cuda(T::Type, x,y,xy, kernel, batchsize)
    A = CUDA.zeros(T, (cld(length(x), batchsize), cld(length(y), batchsize), length(xy)))
    x = convert(CuVector{T}, collect(x))
    y = convert(CuVector{T}, collect(y))
    xy = convert(CuVector{Tuple{T,T}}, xy)
    num_threads = (256,1)
    num_blocks = cld.(size(A)[1:2], num_threads)
    stride = size(A)[1:2]
    @cuda blocks=num_blocks threads=num_threads integrate_cuda_inner!(A, x, y, xy, kernel, stride)
    z = sum(A, dims=(1,2))
    a = collect(z)
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(z)
    a
end
