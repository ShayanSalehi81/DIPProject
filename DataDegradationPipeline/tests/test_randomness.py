from microq_phase1.core.randomness import derive_seed


def test_seed_derivation_is_stable_and_context_sensitive() -> None:
    assert derive_seed(7, "image", "noise") == derive_seed(7, "image", "noise")
    assert derive_seed(7, "image", "noise") != derive_seed(7, "image", "blur")
