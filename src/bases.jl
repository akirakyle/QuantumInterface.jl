abstract type Space end
abstract type HilbertSpace <: Space end

"""
Abstract type for all specialized bases of a Hilbert space.

This type specifies an orthonormal basis for the Hilbert space of the given
system. All subtypes must implement `Base.:(==)` and `dimension`, where the
latter should return the total dimension of the Hilbert space.

Composite systems can be defined with help of [`CompositeBasis`](@ref).
Custom subtypes can also define composite systems by implementing
`Base.length` and `Base.getindex`.

All relevant properties of concrete subtypes of `Basis` defined in
`QuantumInterface` should be accessed using their documented functions and
should not assume anything about the internal representation of instances of
these types (i.e. do not access the fields of the structs directly).
"""
abstract type Basis <: HilbertSpace end
#abstract type StateBasis <: Basis end
#abstract type OperatorBasis <: Basis end

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

"""
    getindex(b::Space)

Get the i'th factor in the tensor product decomposition of the basis into
subsystems.

See also [`TensorSpace`](@ref).
"""
Base.getindex(b::Space, i) = throw(ArgumentError("getindex() is not defined for $(typeof(b))"))
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
# OneModeHilbertSpaces
##

struct FiniteSpace <: HilbertSpace
    N::Int
end
Base.:(==)(s1::FiniteSpace, s2::FiniteSpace) = s1.N == s2.N
dimension(s::FiniteSpace) = s.N

struct InfiniteSpace <: HilbertSpace end
dimension(b::CompositeBasis) = Inf

struct ArbitrarySpace <: HilbertSpace end
dimension(b::CompositeBasis) = NaN

const OneModeHilbertSpace = Union{FiniteSpace, InfiniteSpace, ArbitrarySpace}
Base.length(b::OneModeHilbertSpace) = 1
Base.getindex(b::OneModeHilbertSpace, i) = i==1 ? b : throw(BoundsError(b,i))

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
struct TensorSpace <: HilbertSpace
    spaces::Vector{Space}
    shape::Vector{Int}
    lengths::Vector{Int}
    N::Int
    D::Int
    function TensorSpace(spaces::Vector{S}) where S<:Space
        shape_ = mapreduce(shape, vcat, bases)
        lengths = cumsum(map(length, bases))
        new{B}(bases, shape_, lengths, lengths[end], prod(shape_))
    end
end
TensorSpace(bases::Basis...) = TensorSpace([bases...])
TensorSpace(bases::Tuple) = TensorSpace([bases...])

Base.:(==)(s1::TensorSpace, s2::TensorSpace) = all(((i, j),) -> i == j, zip(s1.spaces, s2.spaces))
Base.length(s::TensorSpace) = s.N

function Base.getindex(b::TensorSpace, i::Integer)
    (i < 1 || i > b.N) && throw(BoundsError(b,i))
    spaces_idx = findfirst(l -> i<=l, b.lengths) 
    inner_idx = i - (spaces_idx == 1 ? 0 : b.lengths[spaces_idx-1])
    b.spaces[spaces_idx][inner_idx]
end
Base.getindex(s::TensorSpace, indices) = [s[i] for i in indices]
shape(s::TensorSpace) = s.shape
dimension(s::TensorSpace) = s.D

"""
    tensor(x::Space, y::Space, z::Space...)

Create a [`TensorSpace`](@ref) from the given bases.

Any given TensorSpace is expanded so that the resulting TensorSpace does not
contains another TensorSpace.
"""
tensor(spaces::Space...) = reduce(tensor, spaces)
tensor(s::Space) = s

tensor(s1::HilbertSpace, s2::HilbertSpace) = TensorSpace([s1, s2])

function tensor(b1::TensorBasis, b2::TensorBasis)
    if typeof(b1.spaces[end]) == typeof(b2.spaces[1])
        t = tensor(b1.spaces[end], b2.spaces[1])
        if !(t isa TensorBasis)
            return TensorBasis([b1.spaces[1:end-1]; t;  b2.spaces[2:end]])
        end
    end
    return TensorBasis([b1.spaces; b2.spaces])
end

function tensor(b1::TensorBasis, b2::Basis)
    if b1.spaces[end] isa typeof(b2)
        t = tensor(b1.spaces[end], b2)
        if !(t isa TensorBasis)
            return TensorBasis([b1.spaces[1:end-1]; t])
        end
    end
    return TensorBasis([b1.spaces; b2])
end

function tensor(b1::Basis, b2::TensorBasis)
    if b2.spaces[1] isa typeof(b1)
        t = tensor(b1, b2.spaces[1])
        if !(t isa TensorBasis)
            return TensorBasis([t; b2[2:end]])
        end
    end
    return TensorBasis([b1; b2.spaces])
end

Base.:^(b::Space, N::Integer) = tensor_pow(b, N)

"""
    SumSpace(b1, b2...)

Similar to [`TensorSpace`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumSpace <: HilbertSpace
    shape::Vector{S}
    spaces::Vector{B}
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
directsum(b1::HilbertSpace, b2::HilbertSpace) = SumSpace([dimension(b1), dimension(b2)], [b1, b2])
directsum(b1::SumSpace, b2::SumSpace) = SumSpace([b1.shape, b2.shape], [b1.bases; b2.bases])
directsum(b1::SumSpace, b2::HilbertSpace) = SumSpace([b1.shape; dimension(b2)], [b1.bases; b2])
directsum(b1::HilbertSpace, b2::SumSpace) = SumSpace([dimension(b1); b2.shape], [b1; b2.bases])
directsum(bases::HilbertSpace...) = reduce(directsum, bases)
directsum(basis::HilbertSpace) = basis

# TODO: what to do about embed for SumBasis?
#embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)

##
# Quantum object spaces
##

struct CNumSpace <: Space end

struct VecSpace{T} <: Space
    s::Space
end
Base.:(==)(s1::VecSpace, s2::VecSpace) = s1.s == s2.s
dimension(s::VecSpace) = dimension(s.s)
const KetSpace = VecSpace{AbstractKet}
const BraSpace = VecSpace{AbstractBra}

struct OpSpace{T} <: Space
    sl::Space
    sr::Space
end
OpSpace{T}(b) = OpSpace{T}(b,b)
Base.:(==)(s1::OpSpace, s2::OpSpace) = s1.sl == s2.sl && s1.sr == s2.sr
dimension(s::OpSpace) = dimension(s.sl)*dimension(s.sr)
space_l(s::OpSpace) = s.sl
space_l(s::OpSpace) = s.sr

const OperatorSpace = OpSpace{AbstractOperator}
const SuperKetSpace = OpSpace{AbstractSuperKet}
const SuperBraSpace = OpSpace{AbstractSuperBra}
const SuperOperatorSpace = OpSpace{AbstractSuperOperator}
const ChoiStateSpace = OpSpace{AbstractChoiState}
const KrausSpace = OpSpace{AbstractKraus}
const StinespringSpace = OpSpace{AbstractStinespring}

const QOSpace = Union{CNumSpace, VecSpace, OpSpace}

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
addible(a::QOSpace, b::QOSpace) = a == b
add_space(a::QOSpace, b::QOSpace) = a

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
multiplicable(a::QOSpace, b::QOSpace) = false

multiplicable(a::KetSpace, b::BraSpace) = true
mul_space(a::KetSpace, b::BraSpace) = OperatorSpace(a.s, b.s)

multiplicable(a::BraSpace, b::KetSpace) = a.b == b.b
mul_space(a::BraSpace, b::KetSpace) = CNumSpace()

multiplicable(a::OperatorSpace, b::KetSpace) = a.sr == b.b
mul_space(a::OperatorSpace, b::KetSpace) = KetSpace(a.sl)

multiplicable(a::BraSpace, b::OperatorSpace) = a.b == b.sl
mul_space(a::BraSpace, b::OperatorSpace) = BraSpace(b.sr)

multiplicable(a::OperatorSpace, b::OperatorSpace) = a.sr == b.sl
mul_space(a::OperatorSpace, b::OperatorSpace) = OpSpace(a.sl, b.sr)

#multiplicable(a::SuperKetSpace, b::SuperBraSpace) = true
#multiplicable(a::SuperBraSpace, b::SuperKetSpace) = a.b == b.b

multiplicable(a::SuperOperatorSpace, b::SuperKetSpace) = a.sr.sr == b.sr && a.sr.sl = b.sl
mul_space(a::SuperOperatorSpace, b::SuperKetSpace) = SuperKetSpace(a.sl.sl, a.sr.sr)

multiplicable(a::SuperBraSpace, b::SuperOperatorSpace) = a.b == b.sl
mul_space(a::SuperBraSpace, b::SuperOperatorSpace) = BraSpace(b.sr)

multiplicable(a::SuperOperatorSpace, b::SuperOperatorSpace) = a.sr == b.sl
mul_space(a::SuperOperatorSpace, b::SuperOperatorSpace) = OpSpace(a.sl, b.sr)

"""
    check_multiplicable(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects are not multiplicable
as determined by `multiplicable(a, b)`.  Disabled by use of
[`@compatiblebases`](@ref) anywhere further up in the call stack.
"""
function check_multiplicable(a, b)
    if BASES_CHECK[] && !multiplicable(a, b)
        throw(IncompatibleBases())
    end
    mul_space(a, b)
end

"""
    reduced(a, indices)

Reduced basis, state or operator on the specified subsystems.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are kept. At least one index must be specified.
"""
function reduced(b::Basis, indices)
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
function ptrace(b::Basis, indices)
    J = [i for i in 1:length(b) if i ∉ indices]
    length(J) > 0 || throw(ArgumentError("Tracing over all indices is not allowed in ptrace."))
    reduced(b, J)
end

_index_complement(b::Basis, indices) = complement(length(b), indices)
reduced(a, indices) = ptrace(a, _index_complement(basis(a), indices))

"""
    permutesystems(a, perm)

Change the ordering of the subsystems of the given object.

For a permutation vector `[2,1,3]` and a given object with basis `[b1, b2, b3]`
this function results in `[b2, b1, b3]`.
"""
function permutesystems(b::Basis, perm)
    (length(b) == length(perm)) || throw(ArgumentError("Must have length(b) == length(perm) in permutesystems"))
    isperm(perm) || throw(ArgumentError("Must pass actual permeutation to permutesystems"))
    tensor(b.bases[perm]...)
end


##
# Common bases
##

struct LabeledBasis <: OneModeBasis
    space::OneModeHilbertSpace
    label::Symbol
end
Base.:(==)(b1::LabeledBasis, b2::LabeledBasis) = b1.space == b2.space && b1.label == b2.label
dimension(b::FiniteSpace) = b.D

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included fock
state (default is 0). Note that the dimension of this basis is `N+1-offset`.
The [`cutoff`](@ref) and [`offset`](@ref) functions can be used to obtain the
respective properties of a given `FockBasis`.
"""
struct FockBasis{T<:Integer} <: StateBasis
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
struct NLevelBasis{T<:Integer} <: StateBasis
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
struct SpinBasis{T<:Integer} <: StateBasis
    spinnumber::Rational{T}
    D::T
    N::T
    function SpinBasis(spinnumber::Rational{T}, N=1) where T
        n = numerator(spinnumber)
        d = denominator(spinnumber)
        d==2 || d==1 || throw(ArgumentError("Can only construct integer or half-integer spin basis"))
        n >= 0 || throw(ArgumentError("Can only construct positive spin basis"))
        D = numerator(spinnumber*2 + 1)
        new{T}(spinnumber, D, N)
    end
end
SpinBasis(spinnumber) = SpinBasis(convert(Rational{Int}, spinnumber))

Base.:(==)(b1::SpinBasis, b2::SpinBasis) = b1.D==b2.D && b1.N == b2.N
Base.length(b::SpinBasis) = b.N
Base.getindex(b::SpinBasis, i) = SpinBasis(b.spinnumber, length(i))
shape(b::SpinBasis) = fill(b.D, b.N)
dimension(b::SpinBasis) = b.D^b.N
function tensor(b1::SpinBasis, b2::SpinBasis)
    if b1.spinnumber == b2.spinnumber
        return SpinBasis(b1.spinnumber, b1.N+b2.N)
    else
        return CompositeBasis([b1, b2])
    end
end

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
    KetBraBasis(BL,BR)

The "Ket-Bra" operator basis is the standard representation for the left and
right bases of superoperators. This basis is formed by "vec'ing" the
outer-product "Ket-Bra" basis for an operator with a left Bra basis and right
Ket basis which practically means flipping the Bra to a Ket. The operator itself
is then represented as a "Super-Bra" in this basis and corresponds to
column-stacking its matrix.
"""
struct KetBraBasis{BL<:Basis, BR<:Basis} <: OperatorBasis
    left::BL
    right::BR
end
KetBraBasis(b::Basis) = KetBraBasis(b,b)
basis_l(b::KetBraBasis) = b.left
basis_r(b::KetBraBasis) = b.right
Base.:(==)(b1::KetBraBasis, b2::KetBraBasis) = (b1.left == b2.left && b1.right == b2.right)
dimension(b::KetBraBasis) = dimension(b.left)*dimension(b.right)
tensor(b1::KetBraBasis, b2::KetBraBasis) = KetBraBasis(tensor(b1.left,b2.left), tensor(b1.right, b2.right))

"""
    ChoiBasis(ref_basis,out_basis)

The Choi basis is used to represent superoperators in the Choi representation
where the `ref_basis` denotes the ancillary reference system with which an input
state will be jointly measured in order to accomplish teleportation simulation
of the channel with the channel's output appearing in the `out_basis` system.
"""
struct ChoiBasis{BL<:Basis, BR<:Basis} <: OperatorBasis
    ref::BL
    out::BR
end
basis_l(b::ChoiBasis) = b.ref
basis_r(b::ChoiBasis) = b.out
Base.:(==)(b1::ChoiBasis, b2::ChoiBasis) = (b1.ref == b2.ref && b1.out == b2.out)
dimension(b::ChoiBasis) = dimension(b.ref)*dimension(b.out)
tensor(b1::ChoiBasis, b2::ChoiBasis) = ChoiBasis(tensor(b1.ref,b2.ref), tensor(b1.out, b2.out))

"""
    PauliBasis(N)

The standard Pauli operator basis for an `N` qubit space. This consists of
tensor products of the Pauli matrices I, X, Y, Z, in that order for each qubit.
The dimension of the basis is 2²ᴺ.
"""
struct PauliBasis{T<:Integer} <: OperatorBasis
    N::T
end
basis_l(b::PauliBasis) = SpinBasis(1//2)^b.N
basis_r(b::PauliBasis) = SpinBasis(1//2)^b.N
Base.:(==)(b1::PauliBasis, b2::PauliBasis) = b1.N == b2.N
dimension(b::PauliBasis) = 4^b.N
tensor(b1::PauliBasis, b2::PauliBasis) = PauliBasis(b1.N+b2.N)

"""
    ChiBasis(N)

The basis for a Chi process matrix, which is just the Choi state in the Pauli
operator basis. However we do not use the `ChoiBasis`, partly to have easier
dispatch on types, and partly because there's no sensible way to distingish
between the "reference" and "output" systems as that information is lost in the
computational to Pauli basis transformation (i.e. two indices into one).

TODO explain better why dimension base is 2, see sec III.E.
"""
struct ChiBasis{T<:Integer} <: OperatorBasis
    Nl::T
    Nr::T
end
basis_l(b::ChiBasis) = SpinBasis(1//2)^b.Nl
basis_r(b::ChiBasis) = SpinBasis(1//2)^b.Nr
Base.:(==)(b1::ChiBasis, b2::ChiBasis) = (b1.Nl == b2.Nl && b1.Nr == b2.Nr)
dimension(b::ChiBasis) = 2^(b.Nl+b.Nr)
tensor(b1::ChiBasis, b2::ChiBasis) = ChiBasis(b1.Nl+b2.Nl, b1.Nr+b2.Nr)

"""
    HWPauliBasis(N)

The Hesienberg-Weyl Pauli operator basis consisting of the N represents the
underlying Hilbert space dimension, not the operator basis dimension. For N>2,
this representes the operator basis formed by the generalized Pauli matrices,
also called the clock and shift matrices. The ordering is the usual one: when
the index is written in base-N and thus has only two digits, the least
significant bit gives powers of Z (the clock matrix), and most significant bit
gives powers of X (the shfit matrix).
"""
struct HWPauliBasis{T<:Integer} <: OperatorBasis
    shape::Vector{T}
end
HWPauliBasis(N::Integer) = HWPauliBasis([N])
basis_l(b::HWPauliBasis) = tensor(NLevelBasis.(b.shape)...)
basis_r(b::HWPauliBasis) = tensor(NLevelBasis.(b.shape)...)
Base.:(==)(b1::HWPauliBasis, b2::HWPauliBasis) = b1.shape == b2.shape
dimension(b::HWPauliBasis) = prod([n^2 for n in b.shape])
tensor(b1::HWPauliBasis, b2::HWPauliBasis) = HWPauliBasis([b1.shape; b2.shape])


##
# show methods
##

function show(stream::IO, x::GenericBasis)
    write(stream, "Basis(dim=$(x.dim))")
end

function show(stream::IO, x::CompositeBasis)
    write(stream, "[")
    for i in 1:length(x.bases)
        show(stream, x.bases[i])
        if i != length(x.bases)
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

function show(stream::IO, x::SumBasis)
    write(stream, "[")
    for i in 1:length(x.bases)
        show(stream, x.bases[i])
        if i != length(x.bases)
            write(stream, " ⊕ ")
        end
    end
    write(stream, "]")
end

function show(stream::IO, x::KetBraBasis)
    write(stream, "KetBra(left=$(x.left), right=$(x.right))")
end

function show(stream::IO, x::PauliBasis)
    write(stream, "Pauli(N=$(x.N))")
end

function show(stream::IO, x::HWPauliBasis)
    write(stream, "Pauli($(x.shape))")
end
