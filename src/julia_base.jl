# Common error messages
arithmetic_unary_error(funcname, x::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. dense() or sparse()."))
arithmetic_binary_error(funcname, a::AbstractOperator, b::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. dense() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))


##
# States
##


==(a::AbstractKet, b::AbstractBra) = false
==(a::AbstractBra, b::AbstractKet) = false
-(a::T) where {T<:StateVector} = T(basis(a), -a.data) # FIXME issue #12
*(a::StateVector, b::Number) = b*a
copy(a::T) where {T<:StateVector} = T(basis(a), copy(a.data)) # FIXME issue #12
length(a::StateVector) = dimension(space(a))
space(a::StateVector) = throw(ArgumentError("space() is not defined for this type of state vector: $(typeof(a))."))
basis(a::AbstractKet) = space(space(a))
basis(a::AbstractBra) = space(space(a))
directsum(x::StateVector...) = reduce(directsum, x)

# Array-like functions
Base.size(x::StateVector) = size(x.data) # FIXME issue #12
@inline Base.axes(x::StateVector) = axes(x.data) # FIXME issue #12
Base.ndims(x::StateVector) = 1
Base.ndims(::Type{<:StateVector}) = 1
Base.eltype(x::StateVector) = eltype(x.data) # FIXME issue #12

# Broadcasting
Base.broadcastable(x::StateVector) = x

Base.adjoint(a::StateVector) = dagger(a)
dagger(a::StateVector) = arithmetic_unary_error("Hermitian conjugate", a)


##
# Operators
##

space(a::AbstractOperator) = throw(ArgumentError("space() is not defined for this type of operator: $(typeof(a))."))
length(a::AbstractOperator) = dimension(space(a))
basis(a::AbstractOperator) = (check_samebases(space_l(space(a)), space_r(space(a))); space_l(space(a)))
directsum(a::AbstractOperator...) = reduce(directsum, a)

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

Base.size(op::AbstractOperator) = (dimension(space_l(op)), dimension(space_r(op)))
function Base.size(op::AbstractOperator, i::Int)
    i < 1 && throw(ErrorException("dimension index is < 1"))
    i > 2 && return 1
    i==1 ? dimension(space_l(space(op))) : dimension(space_r(space(op)))
end

Base.adjoint(a::AbstractOperator) = dagger(a)
dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)

conj(a::AbstractOperator) = arithmetic_unary_error("Complex conjugate", a)
conj!(a::AbstractOperator) = conj(a::AbstractOperator)

ptrace(a::AbstractOperator, index) = arithmetic_unary_error("Partial trace", a)
