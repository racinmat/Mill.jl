"""
struct BagModel{T <: MillModel, U <: MillModel} <: MillModel
im::T
a::Aggregation
bm::U
end

use a `im` model on data in `BagNode`, the uses function `a` to aggregate individual bags,
and finally it uses `bm` model on the output
"""
struct BagModel{T <: MillModel, A, U <: MillModel} <: MillModel
    im::T
    a::A
    bm::U
end

Flux.@treelike BagModel

BagModel(im::MillFunction, a, bm::MillFunction) = BagModel(ArrayModel(im), a, ArrayModel(bm))
BagModel(im::MillModel, a, bm::MillFunction) = BagModel(im, a, ArrayModel(bm))
BagModel(im::MillFunction, a, bm::MillModel) = BagModel(ArrayModel(im), a, bm)
BagModel(im::MillFunction, a) = BagModel(im, a, identity)
BagModel(im::MillModel, a) = BagModel(im, a, ArrayModel(identity))

(m::BagModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::WeightedBagNode) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

function modelprint(io::IO, m::BagModel; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "BagModel\n", color=c)
    paddedprint(io, "  ├── ", color=c, pad=pad)
    modelprint(io, m.im, pad=[pad; (c, "  │   ")])
    paddedprint(io, "  ├── ", color=c, pad=pad)
    modelprint(io, m.a, pad=[pad; (c, "  │   ")])
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.bm, pad=[pad; (c, "  │   ")])
end
