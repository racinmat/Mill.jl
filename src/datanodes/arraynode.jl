using LearnBase
import Base: cat, vcat, hcat

"""


"""
struct ArrayNode{A<:AbstractArray,C} <: AbstractNode
    data::A
    metadata::C
end

ArrayNode(data::AbstractMatrix) = ArrayNode(data, nothing)
# ArrayNode(data::AbstractNode, a...) = data

Flux.@functor ArrayNode

mapdata(f, x::ArrayNode) = ArrayNode(mapdata(f, x.data), x.metadata)

Base.ndims(x::ArrayNode) = Colon()
LearnBase.nobs(a::ArrayNode) = size(a.data, 2)
LearnBase.nobs(a::ArrayNode, ::Type{ObsDim.Last}) = nobs(a)

function reduce(::typeof(catobs), as::Vector{T}) where {T<:ArrayNode}
    data = reduce(catobs, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    ArrayNode(data, metadata)
end

function reduce(::typeof(vcat), as::Vector{T}) where {T<:ArrayNode}
    data = reduce(vcat, [a.data for a in as])
    metadata = as[1].metadata == nothing ? nothing : as[1].metadata
    ArrayNode(data, metadata)
end

# hcat and vcat only for ArrayNode
function Base.vcat(as::ArrayNode...)
    data = vcat([a.data for a in as]...)
    metadata = as[1].metadata == nothing ? nothing : as[1].metadata
    ArrayNode(data, metadata)
end

Base.hcat(as::ArrayNode...) = reduce(catobs, collect(as))

Base.getindex(x::ArrayNode, i::VecOrRange) = ArrayNode(subset(x.data, i), subset(x.metadata, i))

Base.hash(e::ArrayNode{A,C}, h::UInt) where {A,C} = hash((A, C, e.data, e.metadata), h)
Base.:(==)(e1::ArrayNode{A,C}, e2::ArrayNode{A,C}) where {A,C} = e1.data == e2.data && e1.metadata == e2.metadata
