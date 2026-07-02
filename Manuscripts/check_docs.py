"""Lightweight document-level checks for the active manuscript DOCX files."""

from pathlib import Path

from docx import Document

import verify_submission_artifacts as v

OUT = Path(__file__).resolve().parent / "generated"


def main() -> int:
    main_doc = v.inspect_docx(v.MAIN_DOCX, stop_heading="References")
    supp_doc = v.inspect_docx(v.SUPPLEMENT_DOCX)
    document = Document(v.MAIN_DOCX)

    abstract_words = v.count_words(v.section_text(document, "Abstract", "Introduction"))
    main_text_words = v.count_words(v.main_text_for_word_count(document))
    citation_order = v.citation_first_appearance(main_doc["superscript_runs"])

    print(f"main_text_words      = {main_text_words} (limit {v.MAIN_TEXT_WORD_LIMIT})")
    print(f"abstract_words       = {abstract_words} (limit {v.ABSTRACT_WORD_LIMIT})")
    print(f"main_exhibits        = {main_doc['exhibits']} (limit {v.MAIN_EXHIBIT_LIMIT})")
    print(f"main em/semicolon/colon = {main_doc['em_dash_count']}/{main_doc['semicolon_count']}/{main_doc['colon_count']}")
    print(f"supp em/semicolon/colon = {supp_doc['em_dash_count']}/{supp_doc['semicolon_count']}/{supp_doc['colon_count']}")
    print(f"main superscript runs  = {len(main_doc['superscript_runs'])}")
    print(f"supp superscript runs  = {len(supp_doc['superscript_runs'])}")
    print(f"citations out_of_order = {citation_order['out_of_order']}")
    print(f"max cited             = {citation_order['max_cited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
