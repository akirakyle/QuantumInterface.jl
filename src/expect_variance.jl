"""
    expect(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number.
"""
function expect(indices, op::AbstractOperator{KetBraBasis{L,B1,B2}}, state::AbstractOperator{KetBraBasis{M,B3,B3}}) where {L,M,B1,B2,B3<:CompositeBasis}
    N = nsubsystems(state)
    indices_ = complement(N, indices)
    expect(op, ptrace(state, indices_))
end

expect(index::Integer, op::AbstractOperator{KetBraBasis{N,B1,B2}}, state::AbstractOperator{KetBraBasis{M,B3,B3}}) where {N,M,B1,B2,B3<:CompositeBasis} = expect([index], op, state)
expect(op::AbstractOperator, states::Vector) = [expect(op, state) for state=states]
expect(indices, op::AbstractOperator, states::Vector) = [expect(indices, op, state) for state=states]

expect(op::AbstractOperator{KetBraBasis{B1,B2}}, state::AbstractOperator{KetBraBasis{B2,B2}}) where {B1,B2} = tr(op*state)

"""
    variance(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number
"""
function variance(indices, op::AbstractOperator{KetBraBasis{L,B,B}}, state::AbstractOperator{KetBraBasis{M,BC,BC}}) where {L,M,B,BC<:CompositeBasis}
    N = nsubsystems(state)
    indices_ = complement(N, indices)
    variance(op, ptrace(state, indices_))
end

variance(index::Integer, op::AbstractOperator{KetBraBasis{N,B,B}}, state::AbstractOperator{KetBraBasis{M,BC,BC}}) where {N,M,B,BC<:CompositeBasis} = variance([index], op, state)
variance(op::AbstractOperator, states::Vector) = [variance(op, state) for state=states]
variance(indices, op::AbstractOperator, states::Vector) = [variance(indices, op, state) for state=states]

function variance(op::AbstractOperator{KetBraBasis{N,B,B}}, state::AbstractOperator{KetBraBasis{N,B,B}}) where {N,B}
    expect(op*op, state) - expect(op, state)^2
end
