# https://arxiv.org/pdf/1311.1780.pdf
struct SegmentedPNorm{T} <: AggregationFunction
    ρ::T
    c::T
end

SegmentedPNorm(d::Int) = SegmentedPNorm(param(randn(Float32, d)), param(randn(Float32, d)))
Flux.@treelike SegmentedPNorm

p_map(ρ) = 1 .+ log.(1 .+ exp.(ρ))
inv_p_map(p) = log.(exp.(p-1) .- 1)

modelprint(io::IO, n::SegmentedPNorm; pad=[]) = paddedprint(io, "SegmentedPNorm($(length(n.ρ)))")

function segmented_pnorm(x::Matrix, p::Vector, c::Vector, bags::AbstractBags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += abs(x[i, bi] - c[i]) ^ p[i] / length(b)
            end
        end
        o[:, j] .^= 1 ./ p
    end
    o
end

function segmented_pnorm(x::Matrix, p::Vector, c::Vector, bags::AbstractBags, w::Vector)
    @assert all(w .> 0)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x, 1)
                o[i, j] += w[bi] * abs(x[i, bi] - c[i]) ^ p[i] / ws
            end
        end
        o[:, j] .^= 1 ./ p
    end
    o
end

function segmented_pnorm_back(Δ, x::TrackedMatrix, p::Vector, ρ::Vector, c::Vector, bags::AbstractBags, n::Matrix)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zero(x)
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = Δ[i, j] * sign(x[i, bi] - c[i])
                dx[i, bi] /= length(b)
                dx[i, bi] *= (abs(x[i, bi] - c[i]) / n[i, j]) ^ (p[i] - 1)
            end
        end
    end
    dx, nothing, nothing , nothing
end

function segmented_pnorm_back(Δ, x::TrackedMatrix, p::Vector, ρ::Vector, c::Vector, bags::AbstractBags, w::Vector, n::Matrix)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = zero(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = Δ[i, j] * w[bi] * sign(x[i, bi] - c[i])
                dx[i, bi] /= ws
                dx[i, bi] *= (abs(x[i, bi] - c[i]) / n[i, j]) ^ (p[i] - 1)
            end
        end
    end
    dx, nothing, nothing , nothing, nothing
end

function segmented_pnorm_back(Δ, x::TrackedMatrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::AbstractBags, n::Matrix)
    x = Flux.data(x)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dx = zero(x)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dc = zero(c)
    dcs = zero(c)
    @inbounds for (j, b) in enumerate(bags)
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dx[i, bi] = Δ[i, j] * sig
                dx[i, bi] /= length(b)
                dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
                dps1[i] += ab ^ p[i] * log(ab)
                dps2[i] += ab ^ p[i]
                dcs[i] -= sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= max(1, length(b))
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    dx, dρ, dc, nothing

end

function segmented_pnorm_back(Δ, x::TrackedMatrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::AbstractBags, w::Vector, n::Matrix)
    x = Flux.data(x)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dx = zero(x)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dc = zero(c)
    dcs = zero(c)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dx[i, bi] = Δ[i, j] * w[bi] * sig
                dx[i, bi] /= ws
                dx[i, bi] *= (ab / n[i, j]) ^ (p[i] - 1)
                dps1[i] += w[bi] * ab ^ p[i] * log(ab)
                dps2[i] += w[bi] * ab ^ p[i]
                dcs[i] -= w[bi] * sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= ws
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    dx, dρ, dc, nothing, nothing
end

function segmented_pnorm_back(Δ, x::Matrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::AbstractBags, n::Matrix)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dc = zero(c)
    dcs = zero(c)
    @inbounds for (j, b) in enumerate(bags)
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dps1[i] +=  ab ^ p[i] * log(ab)
                dps2[i] +=  ab ^ p[i]
                dcs[i] -= sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(max(1, length(b)))) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= max(1, length(b))
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    nothing, dρ, dc, nothing
end

function segmented_pnorm_back(Δ, x::Matrix, p::TrackedVector, ρ::TrackedVector, c::TrackedVector, bags::AbstractBags, w::Vector, n::Matrix)
    p = Flux.data(p)
    ρ = Flux.data(ρ)
    c = Flux.data(c)
    Δ = Flux.data(Δ)
    dp = zero(p)
    dps1 = zero(p)
    dps2 = zero(p)
    dc = zero(c)
    dcs = zero(c)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        dcs .= 0
        dps1 .= 0
        dps2 .= 0
        for bi in b
            for i in 1:size(x,1)
                ab = abs(x[i, bi] - c[i])
                sig = sign(x[i, bi] - c[i])
                dps1[i] +=  w[bi] * ab ^ p[i] * log(ab)
                dps2[i] +=  w[bi] * ab ^ p[i]
                dcs[i] -= w[bi] * sig * (ab ^ (p[i] - 1))
            end
        end
        t = n[:, j] ./ p .* (dps1 ./ dps2 .- (log.(dps2) .- log(ws)) ./ p)
        dp .+= Δ[:, j] .* t
        dcs ./= ws
        dcs .*= n[:, j] .^ (1 .- p)
        dc .+= Δ[:, j] .* dcs
    end
    dρ = dp .* σ.(ρ)
    nothing, dρ, dc, nothing, nothing
end

(n::SegmentedPNorm)(x, args...) = segmented_pnorm(x, p_map(Flux.data(n.ρ)), Flux.data(n.c), args...)
(n::SegmentedPNorm)(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

(n::SegmentedPNorm{<:TrackedVector})(x::ArrayNode, args...) = mapdata(x -> n(x, args...), x)

# both x and (ρ, c) can be params
(n::SegmentedPNorm{<:AbstractVector})(x::TrackedMatrix, args...) = _pnorm_grad(x, n.ρ, n.c, args...)
(n::SegmentedPNorm{<:TrackedVector})(x, args...) = _pnorm_grad(x, n.ρ, n.c, args...)
(n::SegmentedPNorm{<:TrackedVector})(x::TrackedMatrix, args...) = _pnorm_grad(x, n.ρ, n.c, args...)

_pnorm_grad(x, ρ, c, args...) = Flux.Tracker.track(_pnorm_grad, x, ρ, c, args...)
Flux.Tracker.@grad function _pnorm_grad(x, ρ, c, args...)
    n = segmented_pnorm(Flux.data(x), p_map(Flux.data(ρ)), Flux.data(c), Flux.data.(args)...)
    grad = Δ -> segmented_pnorm_back(Δ, x, p_map(ρ), ρ, c, args..., n)
    n, grad
end
