function _segmented_mean(x::Matrix, bags::Bags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        for bi in b
            for i in 1:size(x, 1)
                o[i,j] += x[i,bi] / length(b)
            end
        end
    end
    o
end

function _segmented_mean(x::Matrix, bags::Bags, w::Vector)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x, 1)
                o[i,j] += w[bi] * x[i,bi] / ws
            end
        end
    end
    o
end

_segmented_mean_back(x::TrackedArray, bags::Bags) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        for bi in b
            for i in 1:size(x,1)
                dx[i,bi] = Δ[i,j] / length(b)
            end
        end
    end
    dx, nothing
end

_segmented_mean_back(x::TrackedArray, bags::Bags, w::Vector) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = w[bi] * Δ[i, j] / ws
            end
        end
    end
    dx, nothing, nothing
end
