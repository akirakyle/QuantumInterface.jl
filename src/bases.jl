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
Base.length(b::Space) = 1

"""
    getindex(b::Space)

Get the i'th factor in the tensor product decomposition of the basis into
subsystems.

See also [`TensorSpace`](@ref).
"""
Base.getindex(b::Space, i) = i==1 ? b : throw(BoundsError(b,i))
Base.firstindex(b::Space) = 1
Base.lastindex(b::Space) = length(b)
Base.iterate(b::Space, state=1) = state > length(b) ? nothing : (b[state], state+1)

"""
    dimension(b::Space)

Total dimension of the Hilbert space.
"""
dimension(b::Space) = throw(ArgumentError("dimension() is not defined for $(typeof(b))"))

"""
    shape(b::Space)

A vector containing the local dimensions of each Hilbert space in its tensor
product decomposition into subsystems.

See also [`CompositeBasis`](@ref).
"""
shape(b::Space) = [dimension(b[i]) for i=1:length(b)]

##
# Hilbert spaces and quantum object spaces
##

struct CNumberSpace <: Space end
dimension(s::CNumberSpace) = 1

struct FiniteSpace <: Space
    N::Int
    function FiniteSpace(n)
        n > 1 || throw(ArgumentError("FiniteSpace must have dimension two or greater. For one dimensional spaces, use CNumberSpace()"))
        new(n)
    end
end
Base.:(==)(s1::FiniteSpace, s2::FiniteSpace) = s1.N == s2.N
dimension(s::FiniteSpace) = s.N

struct InfiniteSpace <: Space end
dimension(b::InfiniteSpace) = Inf

struct ArbitrarySpace <: Space end
dimension(b::ArbitrarySpace) = NaN

struct VecSpace{T} <: Space
    s::Space
end
Base.:(==)(s1::VecSpace, s2::VecSpace) = s1.s == s2.s
dimension(s::VecSpace) = dimension(s.s)
const KetSpace = VecSpace{AbstractKet}
const BraSpace = VecSpace{AbstractBra}
const SuperKetSpace = VecSpace{AbstractSuperKet}
const SuperBraSpace = VecSpace{AbstractSuperBra}
SuperKetSpace(bl,br) = SuperKetSpace(OpSpace(bl,br))
SuperBraSpace(bl,br) = SuperBraSpace(OpSpace(bl,br))
tensor(s1::VecSpace{T}, s2::VecSpace{T}) where {T} = VecSpace{T}(tensor(s1.s, s2.s))

struct OpSpace{T} <: Space
    sl::Space
    sr::Space
end
OpSpace{T}(b) where T = OpSpace{T}(b,b)
Base.:(==)(s1::OpSpace, s2::OpSpace) = s1.sl == s2.sl && s1.sr == s2.sr
dimension(s::OpSpace) = dimension(s.sl)*dimension(s.sr)
space_l(s::OpSpace) = s.sl
space_r(s::OpSpace) = s.sr

const OperatorSpace = OpSpace{AbstractOperator}
const SuperOperatorSpace = OpSpace{AbstractSuperOperator}
const ChoiStateSpace = OpSpace{AbstractChoiState}
const KrausSpace = OpSpace{AbstractKraus}
const StinespringSpace = OpSpace{AbstractStinespring}
tensor(s1::OpSpace{T}, s2::OpSpace{T}) where T = OpSpace{T}(tensor(s1.sl,s2.sl), tensor(s1.sr, s2.sr))

const QOSpace = Union{VecSpace, OpSpace}


##
# TensorSpace, SumSpace
##

"""
    CompositeBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a vector and creates the shape vector directly from the
dimensions of these subbases. Instead of creating a CompositeBasis directly,
`tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` should be used.
"""
struct TensorSpace <: Space
    spaces::Vector{Space}
end
TensorSpace(spaces::Space...) = TensorSpace([spaces...])
TensorSpace(spaces::Tuple) = TensorSpace([spaces...])

Base.:(==)(s1::TensorSpace, s2::TensorSpace) = all(((i, j),) -> i == j, zip(s1.spaces, s2.spaces))
Base.length(s::TensorSpace) = length(s.spaces)
Base.getindex(s::TensorSpace, i) = s.spaces[i]
shape(s::TensorSpace) = dimension.(s.spaces)
dimension(s::TensorSpace) = prod(shape(s))

"""
    tensor(x::Space, y::Space, z::Space...)

Create a [`TensorSpace`](@ref) from the given bases.

Any given TensorSpace is expanded so that the resulting TensorSpace does not
contains another TensorSpace.
"""
tensor(s::Space) = s
tensor(spaces::Space...) = reduce(tensor, spaces)
tensor(s1::Space, s2::Space) = TensorSpace([s1, s2])

tensor(s1::TensorSpace, s2::TensorSpace) = TensorSpace([s1.spaces; s2.spaces])
tensor(s1::TensorSpace, s2::Space) = TensorSpace([s1.spaces; s2])
tensor(s1::Space, s2::TensorSpace) = TensorSpace([s1; s2.spaces])

Base.:^(b::Space, N::Integer) = tensor_pow(b, N)

"""
    SumSpace(b1, b2...)

Similar to [`TensorSpace`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumSpace <: Space
    spaces::Vector{Space}
end
SumSpace(spaces::Space...) = SumSpace([spaces...])
SumSpace(spaces::Tuple) = SumSpace([spaces...])

Base.:(==)(s1::SumSpace, s2::SumSpace) = all(((i, j),) -> i == j, zip(s1.spaces, s2.spaces))
dimension(s::SumSpace) = sum(dimension.(s.spaces))


"""
    nsubspaces(b)

Return the number of subspaces of a [`SumBasis`](@ref) in its direct sum
decomposition.
"""
nsubspaces(b::SumSpace) = length(b.spaces)

"""
    subspace(b, i)

Return the basis for the `i`th subspace of of a [`SumSpace`](@ref).
"""
subspace(b::SumSpace, i) = b.spaces[i]

"""
    directsum(b1::HilbertSpace, b2::HilbertSpace)

Construct the [`SumSpace`](@ref) out of two sub-bases.
"""
directsum(b1::Space, b2::Space) = SumSpace([b1, b2])
directsum(b1::SumSpace, b2::SumSpace) = SumSpace([b1.bases; b2.bases])
directsum(b1::SumSpace, b2::Space) = SumSpace([b1.bases; b2])
directsum(b1::Space, b2::SumSpace) = SumSpace([b1; b2.bases])
directsum(bases::Space...) = reduce(directsum, bases)
directsum(basis::Space) = basis

# TODO: what to do about embed for SumBasis?
#embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)

##
# Basis checks
##

"""
Exception that should be raised for an illegal algebraic operation.
"""
mutable struct IncompatibleSpaces <: Exception end

const SPACES_CHECK = Ref(true)

"""
    @compatiblespaces

Macro to skip checks for compatible spaces. Useful for `*`, `expect` and similar
functions.
"""
macro compatiblespaces(ex)
    return quote
        SPACES_CHECK[] = false
        local val = $(esc(ex))
        SPACES_CHECK[] = true
        val
    end
end

"""
    samespaces(a::Space, a::Space)

Test if two spaces are the same. Equivalant to `==`. See
[`check_samespaces`](@ref).
"""
samespaces(a::Space, b::Space) = a==b

"""
    check_samespaces(a, b)

Throw an [`IncompatibleSpaces`](@ref) error if the spaces are not the same. See
[`samespaces`](@ref).
"""
function check_samespaces(a, b)
    if SPACES_CHECK[] && !samespaces(a, b)
        throw(IncompatibleSpaces())
    end
end

"""
    addible(a, b)

Check if two quantum objects can be added together.
"""
addible(a::Space, b::Space) = a == b
add_space(a::Space, b::Space) = a

"""
    check_addible(a, b)

Throw an [`IncompatibleSpaces`](@ref) error if the objects are not addible as
determined by `addible(a, b)`.  Disabled by use of [`@compatiblespaces`](@ref)
anywhere further up in the call stack.
"""
function check_addible(a, b)
    if SPACES_CHECK[] && !addible(a, b)
        throw(IncompatibleSpaces())
    end
    add_space(a, b)
end

"""
    multiplicable(a, b)

Check if any two quantum objects can be multiplied in the given order.
"""
multiplicable(a::Space, b::Space) = false

multiplicable(a::CNumberSpace, b::CNumberSpace) = true
mul_space(a::CNumberSpace, b::CNumberSpace) = a

multiplicable(a::KetSpace, b::BraSpace) = true
mul_space(a::KetSpace, b::BraSpace) = OpSpace(a.s, b.s)

multiplicable(a::BraSpace, b::KetSpace) = a.s == b.s
mul_space(a::BraSpace, b::KetSpace) = CNumberSpace()

multiplicable(a::OperatorSpace, b::KetSpace) = a.sr == b.s
mul_space(a::OperatorSpace, b::KetSpace) = KetSpace(a.sl)

multiplicable(a::BraSpace, b::OperatorSpace) = a.s == b.sl
mul_space(a::BraSpace, b::OperatorSpace) = BraSpace(b.sr)

multiplicable(a::OperatorSpace, b::OperatorSpace) = a.sr == b.sl
mul_space(a::OperatorSpace, b::OperatorSpace) = OperatorSpace(a.sl, b.sr)

#multiplicable(a::SuperKetSpace, b::SuperBraSpace) = true
#multiplicable(a::SuperBraSpace, b::SuperKetSpace) = a.b == b.b

multiplicable(a::SuperOperatorSpace, b::SuperKetSpace) = a.sr == b.s
mul_space(a::SuperOperatorSpace, b::SuperKetSpace) = SuperKetSpace(a.sl)

multiplicable(a::SuperBraSpace, b::SuperOperatorSpace) = a.s == b.sl
mul_space(a::SuperBraSpace, b::SuperOperatorSpace) = BraSpace(b.sr)

multiplicable(a::SuperOperatorSpace, b::SuperOperatorSpace) = a.sr == b.sl
mul_space(a::SuperOperatorSpace, b::SuperOperatorSpace) = SuperOperatorSpace(a.sl, b.sr)

multiplicable(a::ChoiStateSpace, b::ChoiStateSpace) = a.sr == b.sl
mul_space(a::ChoiStateSpace, b::ChoiStateSpace) = ChoiStateSpace(a.sl, b.sr)

"""
    check_multiplicable(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects are not multiplicable
as determined by `multiplicable(a, b)`.  Disabled by use of
[`@compatiblebases`](@ref) anywhere further up in the call stack.
"""
function check_multiplicable(a, b)
    if BASES_CHECK[] && !multiplicable(a, b)
        throw(IncompatibleSpaces())
    end
    mul_space(a, b)
end

"""
    reduced(a, indices)

Reduced basis, state or operator on the specified subsystems.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are kept. At least one index must be specified.
"""
function reduced(b::Space, indices)
    if length(indices)==0
        throw(ArgumentError("At least one subsystem must be specified in reduced."))
    elseif length(indices)==1
        return b[indices[1]]
    else
        return tensor(b[indices]...)
    end
end

"""
    ptrace(a, indices)

Partial trace of the given basis, state or operator.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are traced out. The number of indices has to be
smaller than the number of subsystems, i.e. it is not allowed to perform a
full trace.
"""
function ptrace(b::Space, indices)
    J = [i for i in 1:length(b) if i ∉ indices]
    length(J) > 0 || throw(ArgumentError("Tracing over all indices is not allowed in ptrace."))
    reduced(b, J)
end

_index_complement(b::Space, indices) = complement(length(b), indices)
reduced(a, indices) = ptrace(a, _index_complement(basis(a), indices))

"""
    permutesystems(a, perm)

Change the ordering of the subsystems of the given object.

For a permutation vector `[2,1,3]` and a given object with basis `[b1, b2, b3]`
this function results in `[b2, b1, b3]`.
"""
function permutesystems(b::Space, perm)
    (length(b) == length(perm)) || throw(ArgumentError("Must have length(b) == length(perm) in permutesystems"))
    isperm(perm) || throw(ArgumentError("Must pass actual permeutation to permutesystems"))
    tensor(b[perm]...)
end


##
# Common bases
##

struct LabeledBasis <: Space
    space::Space
    label::Symbol
end
Base.:(==)(b1::LabeledBasis, b2::LabeledBasis) = b1.space == b2.space && b1.label == b2.label
dimension(b::LabeledBasis) = dimension(b.space)

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included fock
state (default is 0). Note that the dimension of this basis is `N+1-offset`.
The [`cutoff`](@ref) and [`offset`](@ref) functions can be used to obtain the
respective properties of a given `FockBasis`.
"""
struct FockBasis{T<:Integer} <: Basis
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

"""
    cutoff(b::FockBasis)

Return the fock cutoff of the given fock basis.

See [`FockBasis`](@ref).
"""
cutoff(b::FockBasis) = b.N

"""
    offset(b::FockBasis)

Return the offset of the given fock basis.

See [`FockBasis`](@ref).
"""
offset(b::FockBasis) = b.offset


"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{T<:Integer} <: Basis
    N::T
    function NLevelBasis(N::T) where T
        N > 0 || throw(ArgumentError("N must be greater than 0"))
        new{T}(N)
    end
end

Base.:(==)(b1::NLevelBasis, b2::NLevelBasis) = b1.N == b2.N
dimension(b::NLevelBasis) = b.N

"""
    SpinBasis(n, N=1)

Basis for spin-`n` particles over `N` systems.

The basis can be created for arbitrary spin numbers by using a rational number,
e.g. `SpinBasis(3//2)`. The Pauli operators are defined for all possible spin
numbers. The [`spinnumber`](@ref) function can be used to get the spin number
for a `SpinBasis`.
"""
struct SpinBasis{T<:Integer} <: Basis
    spinnumber::Rational{T}
    function SpinBasis(spinnumber::Rational{T}) where T
        n = numerator(spinnumber)
        d = denominator(spinnumber)
        d==2 || d==1 || throw(ArgumentError("Can only construct integer or half-integer spin basis"))
        n >= 0 || throw(ArgumentError("Can only construct positive spin basis"))
        new{T}(spinnumber)
    end
end
SpinBasis(spinnumber) = SpinBasis(convert(Rational{Int}, spinnumber))

Base.:(==)(b1::SpinBasis, b2::SpinBasis) = b1.D==b2.D && b1.N == b2.N
dimension(b::SpinBasis) = numerator(b.spinnumber*2 + 1)

"""
    spinnumber(b::SpinBasis)

Return the spin number of the given spin basis.

See [`SpinBasis`](@ref).
"""
spinnumber(b::SpinBasis) = b.spinnumber


##
# Operator Bases
##


"""
    PauliOpBasis(d)

The standard Pauli operator basis for an `N` qubit space. This consists of
tensor products of the Pauli matrices I, X, Y, Z, in that order for each qubit.
The dimension of the basis is 2²ᴺ.

The Hesienberg-Weyl Pauli operator basis consisting of the N represents the
underlying Hilbert space dimension, not the operator basis dimension. For N>2,
this representes the operator basis formed by the generalized Pauli matrices,
also called the clock and shift matrices. The ordering is the usual one: when
the index is written in base-N and thus has only two digits, the least
significant bit gives powers of Z (the clock matrix), and most significant bit
gives powers of X (the shfit matrix).
"""
struct PauliOpBasis{T<:Integer} <: OperatorBasis
    d::T
end
Base.:(==)(b1::PauliOpBasis, b2::PauliOpBasis) = b1.d == b2.d
dimension(b::PauliOpBasis) = b.d^2

##
# show methods
##

#function show(stream::IO, x::GenericBasis)
#    write(stream, "Basis(dim=$(x.dim))")
#end

function show(stream::IO, x::TensorSpace)
    write(stream, "[")
    for i in 1:length(x.spaces)
        show(stream, x.spaces[i])
        if i != length(x.spaces)
            write(stream, " ⊗ ")
        end
    end
    write(stream, "]")
end

function show(stream::IO, x::SpinBasis)
    d = denominator(x.spinnumber)
    n = numerator(x.spinnumber)
    if d == 1
        write(stream, "Spin($n)")
    else
        write(stream, "Spin($n/$d)")
    end
    if x.N > 1
        write(stream, "^$(x.N)")
    end
end

function show(stream::IO, x::FockBasis)
    if iszero(x.offset)
        write(stream, "Fock(cutoff=$(x.N))")
    else
        write(stream, "Fock(cutoff=$(x.N), offset=$(x.offset))")
    end
end

function show(stream::IO, x::NLevelBasis)
    write(stream, "NLevel(N=$(x.N))")
end

function show(stream::IO, x::SumSpace)
    write(stream, "[")
    for i in 1:length(x.spaces)
        show(stream, x.spaces[i])
        if i != length(x.spaces)
            write(stream, " ⊕ ")
        end
    end
    write(stream, "]")
end

#function show(stream::IO, x::KetBraBasis)
#    write(stream, "KetBra(left=$(x.left), right=$(x.right))")
#end

function show(stream::IO, x::PauliOpBasis)
    write(stream, "Pauli(d=$(x.d))")
end
