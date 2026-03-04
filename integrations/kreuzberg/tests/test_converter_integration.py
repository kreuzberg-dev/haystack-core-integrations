# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for KreuzbergConverter.

These tests call kreuzberg's real extraction APIs against fixture files —
no mocking. Since kreuzberg is local-only (no external API), these do
not require any environment variables or skip conditions.
"""

from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream
from kreuzberg import ChunkingConfig, ExtractionConfig, PageConfig

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_pdf_extraction() -> None:
    """PDF extraction returns a document with real text content."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = result["documents"]

    assert len(docs) == 1
    assert "Sample PDF" in docs[0].content
    assert "Lorem ipsum" in docs[0].content
    assert docs[0].meta["file_path"] == "sample.pdf"
    assert docs[0].meta["mime_type"] == "application/pdf"


@pytest.mark.integration
def test_txt_extraction() -> None:
    """TXT extraction returns content matching the source file."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    docs = result["documents"]

    assert len(docs) == 1
    assert "sample text document for testing the Kreuzberg converter" in docs[0].content
    assert "multiple paragraphs" in docs[0].content


@pytest.mark.integration
def test_docx_extraction() -> None:
    """DOCX extraction returns real text content."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.docx"])
    docs = result["documents"]

    assert len(docs) == 1
    assert "Demonstration of DOCX support" in docs[0].content
    assert docs[0].meta["mime_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.mark.integration
def test_html_extraction() -> None:
    """HTML extraction returns text stripped of markup."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.html"])
    docs = result["documents"]

    assert len(docs) == 1
    assert "Sample Document" in docs[0].content
    assert "sample HTML document for testing the Kreuzberg converter" in docs[0].content
    # Should not contain raw HTML tags
    assert "<h1>" not in docs[0].content

@pytest.mark.integration
def test_multiple_mixed_sources() -> None:
    """All four fixture file types processed, each producing a document."""
    sources = [
        FIXTURES_DIR / "sample.pdf",
        FIXTURES_DIR / "sample.txt",
        FIXTURES_DIR / "sample.docx",
        FIXTURES_DIR / "sample.html",
    ]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources)
    docs = result["documents"]

    assert len(docs) == 4
    assert all(doc.content for doc in docs)


@pytest.mark.integration
def test_batch_vs_sequential_parity() -> None:
    """Batch and sequential extraction produce equivalent content."""
    sources = [
        FIXTURES_DIR / "sample.pdf",
        FIXTURES_DIR / "sample.txt",
        FIXTURES_DIR / "sample.docx",
        FIXTURES_DIR / "sample.html",
    ]
    batch_result = KreuzbergConverter(batch=True).run(sources=sources)
    sequential_result = KreuzbergConverter(batch=False).run(sources=sources)

    batch_docs = batch_result["documents"]
    seq_docs = sequential_result["documents"]

    assert len(batch_docs) == len(seq_docs)
    for b, s in zip(batch_docs, seq_docs):
        assert b.content == s.content


@pytest.mark.integration
def test_metadata_populated_on_real_docs() -> None:
    """Real extraction populates mime_type, output_format, and file_path."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    meta = result["documents"][0].meta

    assert "mime_type" in meta
    assert "output_format" in meta
    assert meta["file_path"] == "sample.pdf"


@pytest.mark.integration
def test_custom_metadata_single_dict() -> None:
    """A single metadata dict is applied to all output documents."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources, meta={"project": "haystack"})
    docs = result["documents"]

    assert len(docs) == 2
    assert all(doc.meta["project"] == "haystack" for doc in docs)


@pytest.mark.integration
def test_custom_metadata_per_source() -> None:
    """Per-source metadata list applies correct dict to each document."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    meta = [{"idx": 0}, {"idx": 1}]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources, meta=meta)
    docs = result["documents"]

    assert docs[0].meta["idx"] == 0
    assert docs[1].meta["idx"] == 1


@pytest.mark.integration
def test_directory_expansion() -> None:
    """Passing a directory processes all files inside it."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR])
    docs = result["documents"]

    # fixtures/ contains 4 files
    assert len(docs) == 4
    assert all(doc.content for doc in docs)


@pytest.mark.integration
def test_raw_extraction_output() -> None:
    """raw_extraction list has one entry per source with content and mime_type."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources)
    raw = result["raw_extraction"]

    assert len(raw) == 2
    assert all("content" in r for r in raw)
    assert all("mime_type" in r for r in raw)
    assert raw[0]["mime_type"] == "application/pdf"


@pytest.mark.integration
def test_per_page_extraction() -> None:
    """Per-page config on a 3-page PDF produces 3 documents."""
    config = ExtractionConfig(pages=PageConfig(extract_pages=True))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = result["documents"]

    assert len(docs) == 3
    assert all(doc.content for doc in docs)
    page_numbers = [doc.meta.get("page_number") for doc in docs]
    assert page_numbers == [1, 2, 3]


@pytest.mark.integration
def test_per_page_raw_extraction_is_one_per_source() -> None:
    """Even with per-page docs, raw_extraction has one entry per source."""
    config = ExtractionConfig(pages=PageConfig(extract_pages=True))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    assert len(result["documents"]) == 3
    assert len(result["raw_extraction"]) == 1


@pytest.mark.integration
def test_chunking_produces_multiple_documents() -> None:
    """Chunking config splits PDF content into multiple documents."""
    config = ExtractionConfig(chunking=ChunkingConfig(preset="sentence"))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = result["documents"]

    assert len(docs) > 1
    assert all(doc.content for doc in docs)
    # Each chunk should have chunk_index metadata
    assert all("chunk_index" in doc.meta for doc in docs)
    assert all("total_chunks" in doc.meta for doc in docs)
    assert docs[0].meta["chunk_index"] == 0


@pytest.mark.integration
def test_bytestream_source() -> None:
    """ByteStream input produces equivalent output to file path input."""
    pdf_path = FIXTURES_DIR / "sample.pdf"
    bytestream = ByteStream(data=pdf_path.read_bytes(), mime_type="application/pdf")

    converter = KreuzbergConverter()
    result = converter.run(sources=[bytestream])
    docs = result["documents"]

    assert len(docs) == 1
    assert "Sample PDF" in docs[0].content
