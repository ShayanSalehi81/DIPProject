from __future__ import annotations

from microq_phase1.domain.models import DegradationKind, Modality, PoolRole, Severity


def test_string_enums_expose_their_values_as_strings() -> None:
    members = (
        PoolRole.TRAINING,
        Modality.FLUORESCENCE,
        Severity.MEDIUM,
        DegradationKind.POISSON_GAUSSIAN,
    )

    for member in members:
        assert isinstance(member, str)
        assert str(member) == member.value
