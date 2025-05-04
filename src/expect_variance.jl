"""
    expect(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number.
"""
expect(indices, op::AbstractOperator, state::AbstractOperator) = expect(op, reduced(state, indices))

expect(index::Integer, op::AbstractOperator, state::AbstractOperator) = expect([index], op, state)

expect(op::AbstractOperator, states::Vector) = [expect(op, state) for state=states]

expect(indices, op::AbstractOperator, states::Vector) = [expect(indices, op, state) for state=states]

expect(op::AbstractOperator, state::AbstractOperator) = (tr(space(op)*space(state)); tr(op*state))

"""
    variance(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number
"""
variance(indices, op::AbstractOperator, state::AbstractOperator) = variance(op, reduced(state, indices))

variance(index::Integer, op::AbstractOperator, state::AbstractOperator) = variance([index], op, state)

variance(op::AbstractOperator, states::Vector) = [variance(op, state) for state=states]

variance(indices, op::AbstractOperator, states::Vector) = [variance(indices, op, state) for state=states]

variance(op::AbstractOperator, state::AbstractOperator) = expect(op*op, state) - expect(op, state)^2
