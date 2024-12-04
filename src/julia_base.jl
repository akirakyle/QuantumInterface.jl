import Base: ==, +, -, *, /, ^, length, size, getindex, one, exp, conj, conj!, transpose, copy

# Common error messages
arithmetic_unary_error(funcname, x::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. dense() or sparse()."))
arithmetic_binary_error(funcname, a::AbstractOperator, b::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. dense() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))


##
# Bases
##

==(b1::T, b2::T) where {T<:Basis} = true
==(b1::Basis, b2::Basis) = false
==(b1::T, b2::T) where {T<:OperatorBasis} = true
==(b1::OperatorBasis, b2::OperatorBasis) = false
==(b1::T, b2::T) where {T<:SuperOperatorBasis} = true
==(b1::SuperOperatorBasis, b2::SuperOperatorBasis) = false

#size(b::CompositeBasis) = length.(b.bases)
length(b::CompositeBasis) = prod(size(b))
length(b::SumBasis) = sum(length.(x.bases))
length(b::GenericBasis{N}) where {N} = N
length(b::NLevelBasis{N}) where N = N
length(b::SpinBasis) = numerator(2*spinnumber(b) + 1)
length(b::SubspaceBasis) = length(basisstates(b))
length(b::ManyBodyBasis) = length(basisstates(b))
length(b::ChargeBasis) = 2*cutoff(b) + 1
length(b::ShiftedChargeBasis) = cutoff_max(b) - cutoff_min(b) + 1
length(b::FockBasis) = cutoff(b) - offset(b) + 1
length(b::PositionBasis{N}) where N = N
length(b::MomentumBasis{N}) where N = N
length(b::CoherentStateBasis{N}) where N = N

size(b::CompositeOperatorBasis) = reduce(((N1,M1), (N2,M2)) -> (N1*N2, M2*M2), size.(b.bases) ; init=(1,1))
size(b::KetBraBasis) = (length(b.left), length(b.right))
size(b::HeisenbergWeylBasis) = (d = dimensions(b); (prod(d),prod(d)))
size(b::PauliBasis) = (n = nsubsystems(b); (2^n, 2^n))
size(b::GaussianBasis) = (c = cutoffs(b); (prod(c),prod(c)))
size(b::KetKetBraBraBasis) = (size(b.left), size(b.right))

#getindex(b::GenericBasis, i) = i==1 ? b : raise BoundsError(b,i)
getindex(b::CompositeBasis, i) = getindex(b.bases, i)
getindex(b::SumBasis, i) = getindex(b.bases, i)
getindex(b::CompositeOperatorBasis, i) = getindex(b.bases, i)
getindex(b::KetBraBasis, i) = (getindex(b.left, i), getindex(b.right, i))
getindex(b::KetKetBraBraBasis, i) = (getindex(b.left, i), getindex(b.right, i))
getindex(b::HeisenbergWeylBasis{dims}) where {dims} = getindex(dims, i)

function Base.:^(b::Basis, N::Integer)
    if N < 1
        throw(ArgumentError("Power of a basis is only defined for positive integers."))
    end
    tensor((b for i=1:N)...)
end


##
# States
##

-(a::T) where {T<:StateVector} = T(a.basis, -a.data) # FIXME
*(a::StateVector, b::Number) = b*a
copy(a::T) where {T<:StateVector} = T(a.basis, copy(a.data)) # FIXME
length(a::StateVector) = length(basis(a))::Int
directsum(x::StateVector...) = reduce(directsum, x)

# Array-like functions
Base.size(x::StateVector) = size(x.data) # FIXME
@inline Base.axes(x::StateVector) = axes(x.data) #FIXME
Base.ndims(x::StateVector) = 1
Base.ndims(::Type{<:StateVector}) = 1
Base.eltype(x::StateVector) = eltype(x.data) # FIXME

# Broadcasting
Base.broadcastable(x::StateVector) = x

Base.adjoint(a::StateVector) = dagger(a)


##
# Operators
##

length(a::AbstractOperator) = prod(size(basis(a)))
transpose(a::AbstractOperator) = arithmetic_unary_error("Transpose", a)
one(x::Union{<:Basis,<:AbstractOperator}) = identityoperator(x)

# Ensure scalar broadcasting
Base.broadcastable(x::AbstractOperator) = Ref(x)

# Arithmetic operations
+(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::AbstractOperator) = addnumbererror()
+(a::AbstractOperator, b::Number) = addnumbererror()
+(a::AbstractOperator) = a

-(a::AbstractOperator) = arithmetic_unary_error("Negation", a)
-(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Subtraction", a, b)
-(a::Number, b::AbstractOperator) = addnumbererror()
-(a::AbstractOperator, b::Number) = addnumbererror()

*(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Multiplication", a, b)
^(a::AbstractOperator, b::Integer) = Base.power_by_squaring(a, b)

"""
    exp(op::AbstractOperator)

Operator exponential.
"""
exp(op::AbstractOperator) = throw(ArgumentError("exp() is not defined for this type of operator: $(typeof(op)).\nTry to convert to dense operator first with dense()."))

Base.size(op::AbstractOperator) = size(basis(op))
function Base.size(op::AbstractOperator, i::Int)
    i < 1 && throw(ErrorException("dimension index is < 1"))
    i > 2 && return 1
    size(basis(op))[i]
end

Base.adjoint(a::AbstractOperator) = dagger(a)

conj(a::AbstractOperator) = arithmetic_unary_error("Complex conjugate", a)
conj!(a::AbstractOperator) = conj(a::AbstractOperator)
