# https://arxiv.org/abs/1511.05286
struct SegmentedLSE{T, U} <: AggregationFunction
    ρ::T
    C::U
end

Flux.@functor SegmentedLSE

SegmentedLSE(d::Int) = SegmentedLSE(randn(Float32, d), zeros(Float32, d))

r_map(ρ) = softplus.(ρ)
inv_r_map = (r) -> max.(r, 0) .+ log1p.(-exp.(-abs.(r)))

(m::SegmentedLSE)(x::MaybeMatrix, bags::AbstractBags, w=nothing) = segmented_lse_forw(x, m.C, r_map(m.ρ), bags)
function (m::SegmentedLSE)(x::AbstractMatrix, bags::AbstractBags, w::AggregationWeights, mask::AbstractVector)
    segmented_lse_forw(x .+ typemin(T) * mask', m.C, r_map(m.ρ), bags)
end

function _lse_precomp(x::AbstractMatrix, r, bags)
    M = zeros(eltype(x), length(r), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if !isempty(b)
            for i in eachindex(r)
                M[i, bi] = r[i] * x[i, first(b)]
            end
            for j in b[2:end]
                for i in eachindex(r)
                    M[i, bi] = max(M[i, bi], r[i] * x[i, j])
                end
            end
        end
    end
    M
end

function _segmented_lse_norm(x::AbstractMatrix, C, r, bags::AbstractBags, M)
    y = zeros(eltype(x), length(r), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                y[i, bi] = C[i]
            end
        else
            for j in b
                for i in eachindex(r)
                    y[i, bi] += exp.(r[i] * x[i, j] - M[i, bi])
                end
            end
            for i in eachindex(r)
                y[i, bi] = (log(y[i, bi]) - log(length(b)) + M[i, bi]) / r[i]
            end
        end
    end
    y
end

segmented_lse_forw(::Missing, C::AbstractVector, r, bags::AbstractBags) = repeat(C, 1, length(bags))
function segmented_lse_forw(x::AbstractMatrix, C, r, bags::AbstractBags)
    M = _lse_precomp(x, r, bags)
    _segmented_lse_norm(x, C, r, bags, M)
end

function segmented_lse_back(Δ, y, x, C, r, bags, M)
    dx = zero(x)
    dC = zero(C)
    dr = zero(r)
    s1 = zeros(eltype(x), length(r))
    s2 = zeros(eltype(x), length(r))

    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(C)
                dC[i] += Δ[i, bi]
            end
        else
            for i in eachindex(r)
                s1[i] = s2[i] = 0
            end
            for j in b
                for i in eachindex(r)
                    e = exp(r[i] * x[i, j] - M[i, bi])
                    s1[i] += e
                    s2[i] += x[i, j] * e
                end
            end
            for j in b
                for i in eachindex(r)
                    dx[i, j] = Δ[i, bi] * exp(r[i] * x[i, j] - M[i, bi]) / s1[i]
                end
            end
            for i in eachindex(r)
                dr[i] += Δ[i, bi] * (s2[i]/s1[i] - y[i, bi]) / r[i]
            end
        end
    end
    dx, dC, dr, nothing, nothing
end

function segmented_lse_back(Δ, ::Missing, C, bags)
    dC = zero(C)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(C)
            dC[i] += Δ[i, bi]
        end
    end
    nothing, dC, nothing, nothing, nothing
end

@adjoint function segmented_lse_forw(x::AbstractMatrix, C::AbstractVector, r::AbstractVector, bags::AbstractBags)
    M = _lse_precomp(x, r, bags)
    y = _segmented_lse_norm(x, C, r, bags, M)
    grad = Δ -> segmented_lse_back(Δ, y, x, C, r, bags, M)
    y, grad
end

@adjoint function segmented_lse_forw(x::Missing, C::AbstractVector, r::AbstractVector, bags::AbstractBags)
    y = segmented_lse_forw(x, C, r, bags)
    grad = Δ -> segmented_lse_back(Δ, x, C, bags)
    y, grad
end
