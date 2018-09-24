function _segmented_max(x::Matrix, bags::Bags)
    o = similar(x, size(x,1), length(bags))
    fill!(o, typemin(eltype(x)))
    @inbounds Threads.@threads for j in 1:length(bags)
        for bi in bags[j]
            for i in 1:size(x, 1)
                o[i, j] = max(o[i, j], x[i, bi])
            end
        end
    end
    o[o.== typemin(eltype(x))] .= 0
    o
end

_segmented_max(x::Matrix, bags::Bags, w::Vector) = _segmented_max(x, bags)

_segmented_max_back(x::TrackedArray, bags::Bags) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zero(x)
    v = [similar(x, size(x,1)) for _ in 1:nthreads()]
    idxs = [zeros(Int, size(x,1)) for _ in 1:nthreads()]
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        t = threadid()
        fill!(v[t], typemin(eltype(x)))
        for bi in b
            for i in 1:size(x, 1)
                if v[t][i] < x[i, bi]
                    idxs[t][i] = bi
                    v[t][i] = x[i, bi]
                end
            end
        end

        for i in 1:size(x, 1)
            dx[i, idxs[t][i]] = Δ[i, j]
        end
    end
    dx, nothing
end

_segmented_max_back(x::TrackedArray, bags::Bags, w::Vector) = Δ -> begin
    tuple(_segmented_max_back(x, bags)(Δ)..., nothing)
end
