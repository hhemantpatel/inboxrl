import re

from inboxrl.plots import _macro, write_numbers


def test_macro_names_are_letters_only():
    for name in ["grpo_v5_s0", "zeroshot_1.5b_v2", "fewshot_0.5b", "rule"]:
        assert re.fullmatch(r"[A-Za-z]+", _macro(name)), name
    assert _macro("grpo_v5_s0") != _macro("grpo_v5_s1")
    assert _macro("zeroshot_1.5b") != _macro("zeroshot_1.5b_v2")


def test_write_numbers_emits_macros_and_seed_means(tmp_path):
    def result(f1):
        return {"event_f1": f1, "event_f1_ci": [f1 - 0.05, f1 + 0.05], "date_accuracy": 0.5, "valid_json_ratio": 1.0,
                "level_f1": {"1": 1.0, "3": 0.25}}

    out = tmp_path / "numbers.tex"
    write_numbers({"grpo_v6_s0": result(0.70), "grpo_v6_s1": result(0.60)}, out)
    text = out.read_text()
    assert r"\newcommand{\nmGrpoVSixSZeroF}{70.0}" in text
    assert r"\newcommand{\nmGrpoVSixSOneLvlThree}{25.0}" in text
    assert r"\newcommand{\nmVSixMeanF}{65.0}" in text
