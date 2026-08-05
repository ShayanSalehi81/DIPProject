from microq_robustness.scoring import MultipleChoiceScorer

OPTIONS = ("light microscopy", "electron microscopy", "fluorescence microscopy", "none")


def test_parses_json_answer() -> None:
    scored = MultipleChoiceScorer().score('{"answer":"B"}', OPTIONS)
    assert scored.predicted_index == 1
    assert scored.scoring_status == "parsed"


def test_parses_explicit_answer_with_explanation() -> None:
    scored = MultipleChoiceScorer().score("The answer is C because of the fluorophores.", OPTIONS)
    assert scored.predicted_index == 2


def test_exact_option_text_is_supported() -> None:
    scored = MultipleChoiceScorer().score("electron microscopy", OPTIONS)
    assert scored.predicted_index == 1


def test_ambiguous_prose_is_not_guessed() -> None:
    scored = MultipleChoiceScorer().score("It could be A or B.", OPTIONS)
    assert scored.predicted_index is None
    assert scored.scoring_status == "unparseable"

