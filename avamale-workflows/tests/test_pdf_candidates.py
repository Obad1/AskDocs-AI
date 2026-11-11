"""Synthetic-only candidate extraction safety tests."""
import pytest
from avamale_suite.pdf_candidates import extract_text_candidates, extract_pdf_candidates


def test_extracts_only_labeled_values_and_sources():
    out = extract_text_candidates([
        "Invoice No: INV-001\nConsignee: Fictional Buyer Ltd\n"
        "Quantity: 500\nPackages: 25\nGross Weight: 1000 kg\nCountry of Origin: Nigeria"
    ])
    assert out["status"] == "human_verification_required"
    assert {x["field"] for x in out["candidates"]} == {
        "document_ref", "consignee", "quantity", "packages",
        "gross_weight_kg", "origin_country"
    }
    assert all(x["page"] == 1 and x["status"] == "needs_human_verification"
               for x in out["candidates"])


def test_conflicts_are_held_with_both_sources():
    out = extract_text_candidates(["Quantity: 500", "Quantity: 480"])
    assert out["ambiguous_fields"] == ["quantity"]
    assert [(x["value"], x["page"]) for x in out["candidates"]] == [
        ("500", 1), ("480", 2)
    ]
    assert all(x["status"] == "ambiguous_requires_human" for x in out["candidates"])


def test_absent_fields_not_invented():
    out = extract_text_candidates(["Unstructured shipping document."])
    assert out["candidates"] == []
    assert len(out["missing_fields"]) == 6


@pytest.mark.parametrize("input_data", [b"<html>not pdf</html>", b"%PDF-" + b"Z" * 2_000_000])
def test_invalid_input_rejected(input_data):
    with pytest.raises(ValueError):
        extract_pdf_candidates(input_data)


def test_oversize_and_page_limit_rejected():
    with pytest.raises(ValueError):
        extract_text_candidates(["z" * 150_001])
    with pytest.raises(ValueError):
        extract_text_candidates(["page"] * 16)
