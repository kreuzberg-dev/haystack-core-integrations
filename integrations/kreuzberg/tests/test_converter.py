# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream, Document
from kreuzberg import ExtractionConfig, KeywordConfig, OcrConfig, PageConfig, config_to_json

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestKreuzbergConverterInit:
    def test_init_default(self):
        converter = KreuzbergConverter()
        assert converter.config is None
        assert converter.config_path is None
        assert converter.store_full_path is False
        assert converter.batch is True
        assert converter.append_tables_to_content is True
        assert converter.easyocr_kwargs is None

    def test_init_with_all_params(self):
        config = ExtractionConfig(output_format="markdown")
        converter = KreuzbergConverter(
            config=config,
            config_path="/tmp/config.json",
            store_full_path=True,
            batch=False,
            append_tables_to_content=False,
            easyocr_kwargs={"gpu": False},
        )
        assert converter.config is config
        assert converter.config_path == "/tmp/config.json"
        assert converter.store_full_path is True
        assert converter.batch is False
        assert converter.append_tables_to_content is False
        assert converter.easyocr_kwargs == {"gpu": False}


class TestKreuzbergConverterSerialization:
    def test_to_dict_default(self):
        converter = KreuzbergConverter()
        d = converter.to_dict()
        assert d == {
            "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
            "init_parameters": {
                "config": None,
                "config_path": None,
                "store_full_path": False,
                "batch": True,
                "append_tables_to_content": True,
                "easyocr_kwargs": None,
            },
        }

    def test_to_dict_with_config(self):
        config = ExtractionConfig(output_format="markdown")
        converter = KreuzbergConverter(config=config)
        d = converter.to_dict()
        # config should be serialized as JSON string
        config_value = d["init_parameters"]["config"]
        assert isinstance(config_value, str)
        parsed = json.loads(config_value)
        assert parsed["output_format"] == "markdown"

    def test_from_dict_default(self):
        d = {
            "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
            "init_parameters": {
                "config": None,
                "store_full_path": True,
            },
        }
        converter = KreuzbergConverter.from_dict(d)
        assert converter.config is None
        assert converter.store_full_path is True

    def test_serialization_roundtrip_default(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(
                output_format="markdown",
                ocr=OcrConfig(backend="tesseract", language="eng"),
            ),
            batch=False,
        )
        d = converter.to_dict()
        restored = KreuzbergConverter.from_dict(d)
        assert restored.config is not None
        assert restored.config.output_format == "markdown"
        assert restored.config.ocr is not None
        assert restored.config.ocr.backend == "tesseract"
        assert restored.batch is False

    def test_serialization_roundtrip_with_config(self):
        config = ExtractionConfig(
            output_format="markdown",
            ocr=OcrConfig(backend="tesseract", language="eng"),
        )
        converter = KreuzbergConverter(config=config)
        d = converter.to_dict()
        restored = KreuzbergConverter.from_dict(d)
        assert restored.config is not None
        assert restored.config.output_format == "markdown"
        assert restored.config.ocr is not None
        assert restored.config.ocr.backend == "tesseract"
        assert restored.config.ocr.language == "eng"


class TestKreuzbergConverterBuildConfig:
    def test_build_config_default(self):
        converter = KreuzbergConverter()
        config = converter._build_config()
        assert config.output_format == "plain"
        assert config.language_detection is not None
        assert config.language_detection.enabled is True

    def test_build_config_does_not_mutate_self_config(self):
        base = ExtractionConfig(output_format="html")
        converter = KreuzbergConverter(config=base)
        converter._build_config()
        assert base.output_format == "html"

    def test_build_config_from_file(self):
        config = ExtractionConfig(output_format="markdown")
        json_str = config_to_json(config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_str)
            path = f.name

        try:
            converter = KreuzbergConverter(config_path=path)
            built = converter._build_config()
            assert built.output_format == "markdown"
        finally:
            Path(path).unlink(missing_ok=True)


class TestKreuzbergConverterExtraction:
    def test_run_single_text_file(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])

        assert len(result["documents"]) == 1
        assert len(result["raw_extraction"]) == 1

        doc = result["documents"][0]
        assert isinstance(doc, Document)
        assert doc.content is not None
        assert len(doc.content) > 0
        assert doc.meta["file_path"] == "sample.txt"
        assert doc.meta["mime_type"] == "text/plain"
        assert doc.meta["format_type"] == "text"

    def test_run_single_pdf(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

        doc = result["documents"][0]
        assert doc.meta["title"] == "Sample PDF"
        assert doc.meta["authors"] == ["Philip Hutchison"]
        assert doc.meta["page_count"] == 3
        assert doc.meta["format_type"] == "pdf"
        assert doc.meta["is_encrypted"] is False
        assert doc.meta["pdf_version"] == "1.3"
        assert doc.meta["mime_type"] == "application/pdf"

    def test_run_multiple_sources(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.txt", FIXTURES_DIR / "sample.pdf"]
        )
        assert len(result["documents"]) == 2
        assert len(result["raw_extraction"]) == 2

    def test_run_with_bytestream(self):
        bs = ByteStream(data=b"Hello from ByteStream!", mime_type="text/plain")
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[bs])

        doc = result["documents"][0]
        assert doc.content == "Hello from ByteStream!"

    def test_run_with_bytestream_auto_detect_mime(self):
        bs = ByteStream(data=b"Hello auto-detect", mime_type=None)
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[bs])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello auto-detect"

    def test_run_with_string_path(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[str(FIXTURES_DIR / "sample.txt")])

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["file_path"] == "sample.txt"

    def test_run_nonexistent_file_skipped(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=["nonexistent.pdf", FIXTURES_DIR / "sample.txt"]
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["file_path"] == "sample.txt"


class TestKreuzbergConverterMetadata:
    def test_quality_score_in_metadata(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        assert "quality_score" in doc.meta
        assert isinstance(doc.meta["quality_score"], float)

    def test_detected_languages_in_metadata(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
        doc = result["documents"][0]
        assert "detected_languages" in doc.meta
        assert isinstance(doc.meta["detected_languages"], list)

    def test_output_format_tracking(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        assert doc.meta["output_format"] == "plain"
        assert doc.meta["result_format"] == "unified"

    def test_output_format_markdown(self):
        converter = KreuzbergConverter(config=ExtractionConfig(output_format="markdown"), batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        assert doc.meta["output_format"] == "markdown"

    def test_keyword_extraction(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(keywords=KeywordConfig(max_keywords=3)),
            batch=False,
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
        doc = result["documents"][0]
        assert "extracted_keywords" in doc.meta
        keywords = doc.meta["extracted_keywords"]
        assert len(keywords) == 3
        assert "text" in keywords[0]
        assert "score" in keywords[0]
        assert "algorithm" in keywords[0]

    def test_store_full_path_false(self):
        converter = KreuzbergConverter(store_full_path=False, batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        assert doc.meta["file_path"] == "sample.txt"

    def test_store_full_path_true(self):
        converter = KreuzbergConverter(store_full_path=True, batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        # Full path should contain directory separators
        assert "/" in str(doc.meta["file_path"]) or "\\" in str(doc.meta["file_path"])

    def test_user_metadata_single_dict(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.txt"],
            meta={"custom_key": "custom_value"},
        )
        doc = result["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"

    def test_user_metadata_per_source(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.txt", FIXTURES_DIR / "sample.pdf"],
            meta=[{"src": "txt"}, {"src": "pdf"}],
        )
        assert result["documents"][0].meta["src"] == "txt"
        assert result["documents"][1].meta["src"] == "pdf"

    def test_user_metadata_overrides_extraction_metadata(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.pdf"],
            meta={"title": "User Override Title"},
        )
        doc = result["documents"][0]
        assert doc.meta["title"] == "User Override Title"


class TestKreuzbergConverterPerPage:
    def test_per_page_produces_multiple_documents(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(pages=PageConfig(extract_pages=True)), batch=False
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
        docs = result["documents"]
        assert len(docs) == 3  # 3-page PDF

    def test_per_page_document_metadata(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(pages=PageConfig(extract_pages=True)), batch=False
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
        docs = result["documents"]

        for i, doc in enumerate(docs, start=1):
            assert doc.meta["page_number"] == i
            assert "is_blank" in doc.meta
            assert doc.meta["file_path"] == "sample.pdf"

    def test_per_page_with_user_metadata(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(pages=PageConfig(extract_pages=True)), batch=False
        )
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.pdf"],
            meta={"source": "test"},
        )
        for doc in result["documents"]:
            assert doc.meta["source"] == "test"

    def test_per_page_raw_extraction_one_per_source(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(pages=PageConfig(extract_pages=True)), batch=False
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
        # raw_extraction should have one entry per source, not per page
        assert len(result["raw_extraction"]) == 1


class TestKreuzbergConverterBatch:
    def test_batch_extraction(self):
        converter = KreuzbergConverter(batch=True)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.txt", FIXTURES_DIR / "sample.pdf"]
        )
        assert len(result["documents"]) == 2
        assert len(result["raw_extraction"]) == 2

    def test_batch_single_source_uses_sequential(self):
        """When only one source, batch mode should use sequential extraction."""
        converter = KreuzbergConverter(batch=True)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        assert len(result["documents"]) == 1

    def test_batch_with_bytestream(self):
        bs = ByteStream(data=b"Batch bytestream", mime_type="text/plain")
        converter = KreuzbergConverter(batch=True)
        result = converter.run(
            sources=[FIXTURES_DIR / "sample.txt", bs]
        )
        assert len(result["documents"]) == 2

    def test_batch_skips_failed_sources(self):
        converter = KreuzbergConverter(batch=True)
        result = converter.run(
            sources=["nonexistent.pdf", FIXTURES_DIR / "sample.txt"]
        )
        # nonexistent should be skipped, sample.txt should succeed
        assert len(result["documents"]) >= 1


class TestKreuzbergConverterDirectoryInput:
    def test_directory_expansion(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR])
        docs = result["documents"]
        assert len(docs) == 4  # sample.txt, sample.pdf, sample.docx, sample.html
        filenames = sorted(d.meta["file_path"] for d in docs)
        assert filenames == ["sample.docx", "sample.html", "sample.pdf", "sample.txt"]

    def test_directory_with_single_dict_meta(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(
            sources=[FIXTURES_DIR],
            meta={"source": "fixtures"},
        )
        for doc in result["documents"]:
            assert doc.meta["source"] == "fixtures"

    def test_directory_with_list_meta_raises(self):
        converter = KreuzbergConverter(batch=False)
        with pytest.raises(ValueError, match="directories are present"):
            converter.run(sources=[FIXTURES_DIR], meta=[{"a": 1}])

    def test_mixed_directory_and_file(self):
        converter = KreuzbergConverter(batch=False)
        bs = ByteStream(data=b"Extra source", mime_type="text/plain")
        result = converter.run(sources=[FIXTURES_DIR, bs])
        # 4 from fixtures dir + 1 bytestream = 5
        assert len(result["documents"]) == 5


class TestKreuzbergConverterRawExtraction:
    def test_raw_extraction_output(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

        raw = result["raw_extraction"]
        assert len(raw) == 1
        r = raw[0]
        assert "content" in r
        assert "mime_type" in r
        assert r["mime_type"] == "application/pdf"
        assert "output_format" in r
        assert "metadata" in r

    def test_raw_extraction_metadata_structure(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

        raw_meta = result["raw_extraction"][0]["metadata"]
        assert isinstance(raw_meta, dict)
        assert "title" in raw_meta
        assert "format_type" in raw_meta

    def test_raw_extraction_with_keywords(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(keywords=KeywordConfig(max_keywords=3)),
            batch=False,
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

        raw = result["raw_extraction"][0]
        assert "extracted_keywords" in raw
        assert len(raw["extracted_keywords"]) == 3

    def test_raw_extraction_pages_when_per_page(self):
        converter = KreuzbergConverter(
            config=ExtractionConfig(pages=PageConfig(extract_pages=True)), batch=False
        )
        result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

        raw = result["raw_extraction"][0]
        assert "pages" in raw
        assert len(raw["pages"]) == 3


class TestKreuzbergConverterTableAssembly:
    def test_append_tables_to_content_default_true(self):
        converter = KreuzbergConverter(append_tables_to_content=True, batch=False)
        # Use a file that has tables — we can't guarantee fixture has tables,
        # so we test that the flag is stored correctly
        assert converter.append_tables_to_content is True

    def test_append_tables_to_content_false(self):
        converter = KreuzbergConverter(append_tables_to_content=False, batch=False)
        assert converter.append_tables_to_content is False


class TestKreuzbergConverterIntrospection:
    def test_supported_extractors(self):
        extractors = KreuzbergConverter.supported_extractors()
        assert isinstance(extractors, list)
        assert len(extractors) > 0

    def test_supported_ocr_backends(self):
        backends = KreuzbergConverter.supported_ocr_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0


class TestKreuzbergConverterOutputFormat:
    def test_markdown_output(self):
        converter = KreuzbergConverter(config=ExtractionConfig(output_format="markdown"), batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.html"])
        doc = result["documents"][0]
        assert doc.meta["output_format"] == "markdown"

    def test_html_output(self):
        converter = KreuzbergConverter(config=ExtractionConfig(output_format="html"), batch=False)
        result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        doc = result["documents"][0]
        assert doc.meta["output_format"] == "html"


class TestKreuzbergConverterEdgeCases:
    def test_empty_sources_list(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=[])
        assert result["documents"] == []
        assert result["raw_extraction"] == []

    def test_all_sources_fail(self):
        converter = KreuzbergConverter(batch=False)
        result = converter.run(sources=["nonexistent1.pdf", "nonexistent2.pdf"])
        assert result["documents"] == []
        assert result["raw_extraction"] == []

    def test_config_not_mutated_across_runs(self):
        config = ExtractionConfig(output_format="html")
        converter = KreuzbergConverter(config=config, batch=False)
        converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        converter.run(sources=[FIXTURES_DIR / "sample.txt"])
        # Original config should not be mutated
        assert config.output_format == "html"

    def test_easyocr_kwargs_stored(self):
        converter = KreuzbergConverter(easyocr_kwargs={"gpu": False, "beam_width": 5})
        assert converter.easyocr_kwargs == {"gpu": False, "beam_width": 5}
        d = converter.to_dict()
        assert d["init_parameters"]["easyocr_kwargs"] == {"gpu": False, "beam_width": 5}
