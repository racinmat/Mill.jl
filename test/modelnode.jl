@testset "testing simple matrix model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ArrayNode(randn(Float32, 4, 5))
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 5)
    @test m isa ArrayModel
    @test eltype(m(x).data) == Float32
end

@testset "testing simple aggregation model" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = BagNode(ArrayNode(randn(Float32, 4, 4)), [1:2, 3:4])
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test m isa BagModel
    @test eltype(m(x).data) == Float32
end

@testset "testing simple tuple models" begin
    layerbuilder(k) = Flux.Dense(k, 2, NNlib.relu)
    x = ProductNode((ArrayNode(randn(Float32, 3, 4)), ArrayNode(randn(Float32, 4, 4))))
    m = reflectinmodel(x, layerbuilder)
    @test eltype(m(x).data) == Float32
    @test size(m(x).data) == (2, 4)
    @test m isa ProductModel
    @test m.ms[1] isa ArrayModel
    @test m.ms[2] isa ArrayModel

    x = ProductNode((BagNode(ArrayNode(randn(Float32, 3, 4)), [1:2, 3:4]),
                  BagNode(ArrayNode(randn(Float32, 4, 4)), [1:1, 2:4])))
    m = reflectinmodel(x, layerbuilder)
    @test size(m(x).data) == (2, 2)
    @test m isa ProductModel
    @test m.ms[1] isa BagModel
    @test m.ms[1].im isa ArrayModel
    @test m.ms[1].bm isa ArrayModel
    @test m.ms[2] isa BagModel
    @test m.ms[2].im isa ArrayModel
    @test m.ms[2].bm isa ArrayModel
end

@testset "testing nested bag model" begin
    bn = BagNode(ArrayNode(randn(Float32, 2, 8)), [1:1, 2:2, 3:6, 7:8])
    x = BagNode(bn, [1:2, 3:4])
    m = reflectinmodel(x, d -> Flux.Dense(d, 2))
    @test size(m(x).data) == (2, 2)
    @test m isa BagModel
    @test m.im isa BagModel
    @test m.im.im isa ArrayModel
    @test m.im.bm isa ArrayModel
    @test m.bm isa ArrayModel
    @test eltype(m(x).data) == Float32

    a = BagNode(BagNode(ArrayNode(randn(2,2)),[1:2]),[1:1])
    b = BagNode(missing,[0:-1])
    c = BagNode(a.data[1:0], [0:-1])
    m = reflectinmodel(a, d -> Dense(d,2), d -> SegmentedMeanMax(d))
    abc = catobs(a, b, c)
    bca = catobs(b, c, a)
    ma = m(a).data
    mb = m(b).data
    mc = m(c).data
    mabc = m(abc).data
    mbca = m(bca).data
    @test mb ≈ mc
    @test mabc[:,1] ≈ ma
    @test mabc[:,2] ≈ mb
    @test mabc[:,3] ≈ mc
    @test mbca[:,1] ≈ mb
    @test mbca[:,2] ≈ mc
    @test mbca[:,3] ≈ ma
end
