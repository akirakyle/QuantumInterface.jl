abstract type Space end
abstract type HilbertSpace end
abstract type OneModeHilbertSpace end

"""
    space(a)

Return the space of a quantum object.
"""
function space end

"""
    length(b::Space)

Return the number of subsystems of a quantum object in its tensor product
decomposition.

See also [`TensorSpace`](@ref).
"""
Base.length(b::Space) = throw(ArgumentError("length() is not defined for $(typeof(b))"))
Base.length(b::OneModeHilbertSpace) = 1

"""
    getindex(b::Space)

Get the i'th factor in the tensor product decomposition of the basis into
subsystems.

See also [`TensorSpace`](@ref).
"""
Base.getindex(b::Space, i) = throw(ArgumentError("getindex() is not defined for $(typeof(b))"))
Base.getindex(b::OneModeHilbertSpace, i) = i==1 ? b : throw(BoundsError(b,i))
Base.firstindex(b::Space) = 1
Base.lastindex(b::Space) = length(b)
Base.iterate(b::Space, state=1) = state > length(b) ? nothing : (b[state], state+1)

"""
    dimension(b::Basis)

Total dimension of the Hilbert space.
"""
dimension(b::Basis) = throw(ArgumentError("dimesion() is not defined for $(typeof(b))"))

"""
    shape(b::Basis)

A vector containing the local dimensions of each Hilbert space in its tensor
product decomposition into subsystems.

See also [`CompositeBasis`](@ref).
"""
shape(b::Basis) = [dimension(b[i]) for i=1:length(b)]

struct FiniteSpace <: OneModeHilbertSpace
    N::Int
end
Base.:(==)(s1::FiniteSpace, s2::FiniteSpace) = s1.N == s2.N
dimension(b::FiniteSpace) = b.D

struct InfiniteSpace <: OneModeHilbertSpace end
dimension(b::CompositeBasis) = Inf

struct ArbitrarySpace <: HilbertSpace end
dimension(b::CompositeBasis) = NaN

struct TensorSpace <: HilbertSpace
    spaces::Vector{Space}
end
TensorSpace(bases::Basis...) = TensorSpace([bases...])
TensorSpace(bases::Tuple) = TensorSpace([bases...])

Base.:(==)(s1::TensorSpace, s2::TensorSpace) = all(((i, j),) -> i == j, zip(s1.spaces, s2.spaces))
Base.length(b::TensorSpace) = length(b.spaces)
Base.getindex(b::TensorSpace, i) = b.spaces[i]
dimension(b::CompositeBasis) = prod(dimension.(b.spaces))

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(spaces::Space...) = reduce(tensor, spaces)
tensor(s::Space) = s

tensor(s1::HilbertSpace, s2::HilbertSpace) = TensorSpace([s1, s2])
tensor(s1::TensorSpace, s2::TensorSpace) = TensorSpace([s1.spaces; s2.spaces])
tensor(s1::TensorSpace, s2::HilbertSpace) = TensorSpace([s1.spaces; s2])
tensor(s1::HilbertSpace, s2::TensorSpace) = TensorSpace([s1; s2.spaces])

Base.:^(s::Space, N::Integer) = tensor_pow(s, N)

"""
    SumBasis(b1, b2...)

Similar to [`CompositeBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumSpace <: HilbertSpace
    spaces::Vector{Space}
end
SumSpace(spaces) = SumSpace([dimension(b) for b in spaces], spaces)
SumSpace(spaces::Space...) = SumSpace([spaces...])
SumSpace(spaces::Tuple) = SumSpace([spaces...])

Base.:(==)(s1::SumSpace, s2::SumSpace) = all(((i, j),) -> i == j, zip(s1.spaces, s2.spaces))
dimension(s::SumSpace) = sum(dimension.(s.spaces))

"""
    nsubspaces(b)

Return the number of subspaces of a [`SumBasis`](@ref) in its direct sum
decomposition.
"""
nsubspaces(s::SumBasis) = length(s.spaces)

"""
    subspace(b, i)

Return the basis for the `i`th subspace of of a [`SumBasis`](@ref).
"""
subspace(b::SumBasis, i) = b.spaces[i]

"""
    directsum(b1::HilbertSpace, b2::HilbertSpace)

Construct the [`SumSpace`](@ref) out of two sub-bases.
"""
directsum(b1::HilbertSpace, b2::HilbertSpace) = SumSpace([dimension(b1), dimension(b2)], [b1, b2])
directsum(b1::SumSpace, b2::SumSpace) = SumSpace([b1.shape, b2.shape], [b1.bases; b2.bases])
directsum(b1::SumSpace, b2::HilbertSpace) = SumSpace([b1.shape; dimension(b2)], [b1.bases; b2])
directsum(b1::HilbertSpace, b2::SumSpace) = SumSpace([dimension(b1); b2.shape], [b1; b2.bases])
directsum(bases::HilbertSpace...) = reduce(directsum, bases)
directsum(basis::HilbertSpace) = basis

"""
struct KetSpace <: Space
    s::Space
end
struct BraSpace <: Space
    s::Space
end
const StateVectorSpace = Union{KetSpace, BraSpace}
Base.:(==)(s1::StateVectorSpace, s2::StateVectorSpace) = s1.s == s2.s
dimension(s::StateVectorSpace) = dimension(s.s)

struct OperatorSpace <: Space
    sl::Space
    sr::Space
end
struct SuperKetSpace <: Space
    sl::Space
    sr::Space
end
struct SuperBraSpace <: Space
    sl::Space
    sr::Space
end
OperatorSpace(b) = OperatorSpace(b,b)
SuperKetSpace(b) = SuperKetSpace(b,b)
const OpAndVecSpace = Union{OperatorSpace, SuperKetSpace, SuperBraSpace}

Base.:(==)(s1::OpAndVecSpace, s2::OpAndVecSpace) = s1.sl == s2.sl && s1.sr == s2.sr
dimension(s::OpAndVecSpace) = dimension(s.sl)*dimension(s.sr)
"""

struct OneSpace{T} <: Space
    s::Space
end
Base.:(==)(s1::OneSpace, s2::OneSpace) = s1.s == s2.s
dimension(s::OneSpace) = dimension(s.s)
const KetSpace = OneSpace{AbstractKet}
const BraSpace = OneSpace{AbstractBra}

struct TwoSpace{T} <: Space
    sl::Space
    sr::Space
end
TwoSpace{T}(b) = TwoSpace{T}(b,b)
Base.:(==)(s1::TwoSpace, s2::TwoSpace) = s1.sl == s2.sl && s1.sr == s2.sr
dimension(s::TwoSpace) = dimension(s.sl)*dimension(s.sr)
space_l(s::TwoSpace) = s.sl
space_l(s::TwoSpace) = s.sr

const OperatorSpace = TwoSpace{AbstractOperator}
const SuperKetSpace = TwoSpace{AbstractSuperKet}
const SuperBraSpace = TwoSpace{AbstractSuperBra}
const SuperOperatorSpace = TwoSpace{AbstractSuperOperator}
const ChoiStateSpace = TwoSpace{AbstractChoiState}
const KrausSpace = TwoSpace{AbstractKraus}
const StinespringSpace = TwoSpace{AbstractStinespring}

abstract type OneModeBasis <: OneModeHilbertSpace end

struct LabeledBasis <: OneModeBasis
    space::OneModeHilbertSpace
    label::Symbol
end
Base.:(==)(b1::LabeledBasis, b2::LabeledBasis) = b1.space == b2.space && b1.label == b2.label
dimension(b::FiniteSpace) = b.D

struct FockBasis{T<:Integer} <: OneModeBasis
    N::T
    offset::T
    function FockBasis(N::T,offset::T=0) where T
        if N < 0 || offset < 0 || N <= offset
            throw(ArgumentError("Fock cutoff and offset must be positive and cutoff must be less than offset"))
        end
        new{T}(N, offset)
    end
end

Base.:(==)(b1::FockBasis, b2::FockBasis) = (b1.N==b2.N && b1.offset==b2.offset)
dimension(b::FockBasis) = b.N - b.offset + 1


