"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(b::Basis) = CompositeBasis(b)
tensor(b1::Basis, b2::Basis) = CompositeBasis(b1, b2)
tensor(b1::CompositeBasis, b2::CompositeBasis) = CompositeBasis(b1.bases..., b2.bases...)
tensor(b1::CompositeBasis, b2::Basis) = CompositeBasis(b1.bases..., b2)
tensor(b1::Basis, b2::CompositeBasis) = CompositeBasis(b1, b2.bases...)
tensor(bases::Basis...) = reduce(tensor, bases)

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(b::OperatorBasis) = CompositeOperatorBasis(b)
tensor(b1::OperatorBasis, b2::OperatorBasis) = CompositeOperatorBasis(b1, b2)
tensor(b1::CompositeOperatorBasis, b2::CompositeOperatorBasis) = CompositeOperatorBasis(b1.bases..., b1.bases...)
tensor(b1::CompositeOperatorBasis, b2::OperatorBasis) = CompositeOperatorBasis(b1.bases..., b2)
tensor(b1::OperatorBasis, b2::CompositeOperatorBasis) = CompositeOperatorBasis(b1, b2.bases...)
tensor(bases::OperatorBasis...) = reduce(tensor, bases)

tensor(b::KetBraBasis) = b # TODO is this right?
tensor(b1::KetBraBasis, b2::KetBraBasis) = KetBraBasis(b1.left⊗b2.left, b1.right⊗b2.right)
tensor(bases::KetBraBasis...) = reduce(tensor, bases)

tensor(b::HeisenbergWeylBasis) = b # TODO is this right?
tensor(b1::HeisenbergWeylBasis, b2::HeisenbergWeylBasis) = HeisenbergWeylBasis(b1.dims..., b2.dims...)
tensor(bases::HeisenbergWeylBasis...) = reduce(tensor, bases)

tensor(b::PauliBasis) = b # TODO is this right?
tensor(b1::PauliBasis, b2::PauliBasis) = PauliBasis(b1.modes+b2.modes)
tensor(bases::PauliBasis...) = reduce(tensor, bases)

tensor(b::GaussianBasis) = b # TODO is this right?
tensor(b1::GaussianBasis, b2::GaussianBasis) = GaussianBasis(b1.dims..., b2.dims...)
tensor(bases::GaussianBasis...) = reduce(tensor, bases)

tensor(b::KetKetBraBraBasis) = b # TODO is this right?
tensor(b1::KetKetBraBraBasis, b2::KetKetBraBraBasis) = KetKetBraBraBasis(b1.left⊗b2.left, b1.right⊗b2.right)
tensor(bases::KetKetBraBraBasis...) = reduce(tensor, bases)


"""
    tensor(x::AbstractOperator, y::AbstractOperator, z::AbstractOperator...)

Tensor product ``\\hat{x}⊗\\hat{y}⊗\\hat{z}⊗…`` of the given operators.
"""
tensor(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Tensor product", a, b)
tensor(op::AbstractOperator) = op
tensor(operators::AbstractOperator...) = reduce(tensor, operators)
tensor(state::StateVector) = state
tensor(states::Vector{T}) where T<:StateVector = reduce(tensor, states)

"""
    directsum(b1::Basis, b2::Basis)

Construct the [`SumBasis`](@ref) out of two sub-bases.
"""
directsum(b::Basis) = CompositeBasis(b)
directsum(b1::Basis, b2::Basis) = CompositeBasis(b1, b2)
directsum(b1::SumBasis, b2::SumBasis) = CompositeBasis(b1.bases..., b2.bases...)
directsum(b1::SumBasis, b2::Basis) = CompositeBasis(b1.bases..., b2)
directsum(b1::Basis, b2::SumBasis) = CompositeBasis(b1, b2.bases...)
directsum(bases::Basis...) = reduce(dicectsum, bases)
