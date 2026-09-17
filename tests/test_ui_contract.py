from pathlib import Path


def _html():
    return (Path(__file__).resolve().parents[1] / "app" / "index.html").read_text(encoding="utf-8")


def test_ui_contains_beginner_and_medical_context_sections():
    html = _html()
    required = [
        "Your Model Assessment",
        "What does this result mean?",
        "Why did the model produce this result?",
        "Medical context:",
        "Clinical stage / severity",
        "Not estimated by FedFairChain",
        "What can you do next?",
        "General Guidance & Safety",
        "There is no universal “safe percentage”",
        "model decision cutoff — not a medical safety limit",
        "Optional: technical model math",
        "Print Results",
    ]
    for text in required:
        assert text in html


def test_ui_does_not_show_end_of_page_how_it_works_sections():
    html = _html()
    assert "How to Read These Results" not in html
    assert "About FedFairChain" not in html
    assert "Simulated federated training architecture" not in html


def test_ui_medical_context_is_non_diagnostic():
    html = _html()
    assert "This output is not a medical diagnosis" in html
    assert "not a clinical guideline" in html
    assert "does not diagnose disease" in html
    assert "Medical Verdict" not in html
    assert "Your true disease risk" not in html


def test_technical_details_are_optional_and_not_per_factor():
    html = _html()
    assert "Show technical details" not in html
    assert html.count("Optional: technical model math") == 1
