# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/
struct LSE{T}
    p::T
end

LSE(d::Int) = LSE(param(randn(d)))
Flux.@treelike LSE

Base.show(io::IO, n::LSE{T}) where T = println(io, "LogSumExp{$T}($(length(n.p)))")

function _segmented_lse(x::Matrix, p::Vector, bags::Bags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += exp(p[i] * x[i, bi])
            end
        end
        o[:, j] .= (log.(o[:, j]) .- log(length(b)))
    end
    o ./ p
end

_segmented_lse(x::Matrix, p::Vector, bags::Bags, w::Vector) = _segmented_lse(x, p, bags)

_segmented_lse_back(x::TrackedArray, p::Vector, bags::Bags, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zero(x)
    ss = [zero(p) for _ in 1:nthreads()]
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        t = threadid()
        ss[t] .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                dx[i, bi] = Δ[i, j] * e
                ss[t][i] += e
            end
        end
        dx[:, b] ./= ss[t]
    end
    dx, nothing, nothing
end

_segmented_lse_back(x::TrackedArray, p::Vector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    tuple(_segmented_lse_back(x, p, bags, n)(Δ)..., nothing)
end

_segmented_lse_back(x::TrackedArray, p::TrackedVector, bags::Bags, n::Matrix) = Δ -> begin
    x = Flux.data(x)
    p = Flux.data(p)
    Δ = Flux.data(Δ)
    dx = zero(x)
    dp = [zero(p) for _ in 1:nthreads()]
    ss1 = [zero(p) for _ in 1:nthreads()]
    ss2 = [zero(p) for _ in 1:nthreads()]
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        t = threadid()
        ss1[t] .= ss2[t] .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                dx[i, bi] = Δ[i, j] * e
                ss1[t][i] += e
                ss2[t][i] += x[i, bi] * e
            end
        end
        dx[:, b] ./= ss1[t]
        dp[t] .+= Δ[:, j] .* (ss2[t] ./ ss1[t] - n[:, j])
    end
    dx, reduce(+, dp) ./ p, nothing
end

_segmented_lse_back(x::TrackedArray, p::TrackedVector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    tuple(_segmented_lse_back(x, p, bags, n)(Δ)..., nothing)
end

_segmented_lse_back(x::Matrix, p::TrackedVector, bags::Bags, n::Matrix) = Δ -> begin
    p = Flux.data(p)
    Δ = Flux.data(Δ)
    dp = [zero(p) for _ in 1:nthreads()]
    ss1 = [zero(p) for _ in 1:nthreads()]
    ss2 = [zero(p) for _ in 1:nthreads()]
    @inbounds Threads.@threads for j in 1:length(bags)
        b = bags[j]
        t = threadid()
        ss1[t] .= ss2[t] .= 0
        for bi in b
            for i in 1:size(x,1)
                e = exp(p[i] * x[i, bi])
                ss1[t][i] += e
                ss2[t][i] += x[i, bi] * e
            end
        end
        dp[t] .+= Δ[:, j] .* (ss2[t] ./ ss1[t] - n[:, j])
    end
    nothing, reduce(+, dp) ./ p, nothing, nothing
end

_segmented_lse_back(x::Matrix, p::TrackedVector, bags::Bags, w::Vector, n::Matrix) = Δ -> begin
    tuple(_segmented_lse_back(x, p, bags, n)(Δ)..., nothing)
end

(n::LSE)(x, args...) = let m = maximum(x, dims=2)
    m .+ _segmented_lse(x .- m, n.p, args...)
end

(n::LSE)(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)
(n::LSE{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

# both x and p can be params
(n::LSE{<:AbstractVector})(x::TrackedArray, args...) = _lse_grad(x, n.p, args...)
(n::LSE{<:TrackedVector})(x, args...) = _lse_grad(x, n.p, args...)
(n::LSE{<:TrackedVector})(x::TrackedArray, args...) = _lse_grad(x, n.p, args...)

_lse_grad(x, p, args...) = let m = maximum(x, dims=2)
    m .+ Flux.Tracker.track(_lse_grad, x .- m, p, args...)
end

Flux.Tracker.@grad function _lse_grad(x, p, args...)
    n = _segmented_lse(Flux.data(x), Flux.data(p), Flux.data.(args)...)
    grad = _segmented_lse_back(x, p, args..., n)
    n, grad
end
