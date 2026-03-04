# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from kreuzberg import ExtractionConfig, ExtractionResult, OcrConfig, config_to_json

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter
from haystack_integrations.components.converters.kreuzberg.converter import _serialize_page_tables, _serialize_warnings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONVERTER_MODULE = "haystack_integrations.components.converters.kreuzberg.converter"


@pytest.mark.unit
def test_init_default() -> None:
    converter = KreuzbergConverter()
    assert converter.config is None
    assert converter.config_path is None
    assert converter.store_full_path is False
    assert converter.batch is True
    assert converter.easyocr_kwargs is None


@pytest.mark.unit
def test_init_with_all_params() -> None:
    config = ExtractionConfig(output_format="markdown")
    converter = KreuzbergConverter(
        config=config,
        config_path="/tmp/config.json",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": False},
    )
    assert converter.config is config
    assert converter.config_path == "/tmp/config.json"
    assert converter.store_full_path is True
    assert converter.batch is False
    assert converter.easyocr_kwargs == {"gpu": False}


@pytest.mark.unit
def test_serialization_to_dict_default() -> None:
    converter = KreuzbergConverter()
    d = converter.to_dict()
    assert d == {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": None,
            "config_path": None,
            "store_full_path": False,
            "batch": True,
            "easyocr_kwargs": None,
        },
    }


@pytest.mark.unit
def test_serialization_to_dict_with_config() -> None:
    config = ExtractionConfig(output_format="markdown")
    converter = KreuzbergConverter(config=config)
    d = converter.to_dict()
    # config should be serialized as JSON string
    config_value = d["init_parameters"]["config"]
    assert isinstance(config_value, str)
    parsed = json.loads(config_value)
    assert parsed["output_format"] == "markdown"


@pytest.mark.unit
def test_serialization_from_dict_default() -> None:
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


@pytest.mark.unit
def test_serialization_roundtrip_with_config() -> None:
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


@pytest.mark.unit
def test_serialization_to_dict_all_non_default_params() -> None:
    config = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(
        config=config,
        config_path="/tmp/config.toml",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": True, "beam_width": 10},
    )
    d = converter.to_dict()
    params = d["init_parameters"]
    assert isinstance(params["config"], str)
    assert json.loads(params["config"])["output_format"] == "html"
    assert params["config_path"] == "/tmp/config.toml"
    assert params["store_full_path"] is True
    assert params["batch"] is False
    assert params["easyocr_kwargs"] == {"gpu": True, "beam_width": 10}


@pytest.mark.unit
def test_serialization_to_dict_with_config_path() -> None:
    converter = KreuzbergConverter(config_path="/some/path/config.yaml")
    d = converter.to_dict()
    assert d["init_parameters"]["config_path"] == "/some/path/config.yaml"


@pytest.mark.unit
def test_serialization_to_dict_config_path_from_path_object() -> None:
    converter = KreuzbergConverter(config_path=Path("/some/path/config.json"))
    d = converter.to_dict()
    # Path should be stored as str
    assert d["init_parameters"]["config_path"] == "/some/path/config.json"
    assert isinstance(d["init_parameters"]["config_path"], str)


@pytest.mark.unit
def test_serialization_from_dict_with_config_json_string() -> None:
    config_json = config_to_json(ExtractionConfig(output_format="markdown"))
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": config_json,
        },
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is not None
    assert converter.config.output_format == "markdown"


@pytest.mark.unit
def test_serialization_from_dict_all_params() -> None:
    config_json = config_to_json(ExtractionConfig(output_format="html", ocr=OcrConfig(backend="tesseract")))
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": config_json,
            "config_path": "/tmp/config.toml",
            "store_full_path": True,
            "batch": False,
            "easyocr_kwargs": {"gpu": False},
        },
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is not None
    assert converter.config.output_format == "html"
    assert converter.config.ocr.backend == "tesseract"
    assert converter.config_path == "/tmp/config.toml"
    assert converter.store_full_path is True
    assert converter.batch is False
    assert converter.easyocr_kwargs == {"gpu": False}


@pytest.mark.unit
def test_serialization_from_dict_empty_init_parameters() -> None:
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {},
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is None
    assert converter.config_path is None
    assert converter.store_full_path is False
    assert converter.batch is True
    assert converter.easyocr_kwargs is None



@pytest.mark.unit
def test_serialization_roundtrip_easyocr_kwargs() -> None:
    converter = KreuzbergConverter(easyocr_kwargs={"gpu": False, "beam_width": 5})
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.easyocr_kwargs == {"gpu": False, "beam_width": 5}


@pytest.mark.unit
def test_serialization_roundtrip_config_path() -> None:
    converter = KreuzbergConverter(config_path="/tmp/kreuzberg.toml")
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config_path == "/tmp/kreuzberg.toml"


@pytest.mark.unit
def test_serialization_roundtrip_all_non_default_params() -> None:
    converter = KreuzbergConverter(
        config=ExtractionConfig(
            output_format="html",
            ocr=OcrConfig(backend="tesseract", language="deu"),
        ),
        config_path="/tmp/fallback.yaml",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": True, "beam_width": 3},
    )
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config.output_format == "html"
    assert restored.config.ocr.backend == "tesseract"
    assert restored.config.ocr.language == "deu"
    assert restored.config_path == "/tmp/fallback.yaml"
    assert restored.store_full_path is True
    assert restored.batch is False
    assert restored.easyocr_kwargs == {"gpu": True, "beam_width": 3}


@pytest.mark.unit
def test_serialization_roundtrip_preserves_to_dict_equality() -> None:
    converter = KreuzbergConverter(
        config=ExtractionConfig(output_format="markdown"),
        store_full_path=True,
        batch=False,
    )
    d1 = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d1)
    d2 = restored.to_dict()
    # Compare non-config params directly
    p1 = {k: v for k, v in d1["init_parameters"].items() if k != "config"}
    p2 = {k: v for k, v in d2["init_parameters"].items() if k != "config"}
    assert p1 == p2
    # Compare config semantically (JSON round-trip may differ in repr)
    c1 = d1["init_parameters"]["config"]
    c2 = d2["init_parameters"]["config"]
    assert json.loads(c1 if isinstance(c1, str) else config_to_json(c1)) == json.loads(
        c2 if isinstance(c2, str) else config_to_json(c2)
    )


@pytest.mark.unit
def test_build_config_default() -> None:
    converter = KreuzbergConverter()
    config = converter._build_config()
    assert config.output_format == "plain"
    assert config.language_detection is not None
    assert config.language_detection.enabled is True


@pytest.mark.unit
def test_build_config_does_not_mutate_self_config() -> None:
    base = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(config=base)
    converter._build_config()
    assert base.output_format == "html"


@pytest.mark.unit
def test_build_config_from_file() -> None:
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


@pytest.mark.unit
def test_build_config_merges_config_and_config_path() -> None:
    file_config = ExtractionConfig(output_format="html")
    json_str = config_to_json(file_config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        path = f.name

    try:
        converter = KreuzbergConverter(
            config=ExtractionConfig(output_format="markdown"),
            config_path=path,
        )
        built = converter._build_config()
        # Explicit config takes priority over file config
        assert built.output_format == "markdown"
        # Language detection auto-enabled
        assert built.language_detection is not None
        assert built.language_detection.enabled is True
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
def test_table_assembly_appends_markdown_to_content() -> None:
    table = MagicMock()
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text\n\n| A | B |\n|---|---|\n| 1 | 2 |"


@pytest.mark.unit
def test_table_assembly_appends_multiple_tables() -> None:
    t1 = MagicMock()
    t1.markdown = "| A |\n|---|\n| 1 |"
    t2 = MagicMock()
    t2.markdown = "| B |\n|---|\n| 2 |"
    content = KreuzbergConverter._assemble_content("Text", [t1, t2], "plain")
    assert content == "Text\n\n| A |\n|---|\n| 1 |\n\n| B |\n|---|\n| 2 |"


@pytest.mark.unit
def test_table_assembly_skips_tables_with_empty_markdown() -> None:
    table = MagicMock()
    table.markdown = ""
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text"


@pytest.mark.unit
def test_table_assembly_no_tables_returns_text_unchanged() -> None:
    assert KreuzbergConverter._assemble_content("text", None, "plain") == "text"
    assert KreuzbergConverter._assemble_content("text", [], "plain") == "text"


@pytest.mark.unit
def test_table_assembly_skipped_for_markdown_format() -> None:
    table = MagicMock()
    table.markdown = "| A |"
    assert KreuzbergConverter._assemble_content("text", [table], "markdown") == "text"


@pytest.mark.unit
def test_table_assembly_skipped_for_html_format() -> None:
    table = MagicMock()
    table.markdown = "| A |"
    assert KreuzbergConverter._assemble_content("text", [table], "html") == "text"


@pytest.mark.unit
def test_introspection_supported_extractors() -> None:
    extractors = KreuzbergConverter.supported_extractors()
    assert isinstance(extractors, list)


@pytest.mark.unit
def test_introspection_supported_ocr_backends() -> None:
    backends = KreuzbergConverter.supported_ocr_backends()
    assert isinstance(backends, list)
    assert len(backends) > 0


@pytest.mark.unit
def test_edge_empty_sources_list(sequential_converter: KreuzbergConverter) -> None:
    converter = sequential_converter
    result = converter.run(sources=[])
    assert result["documents"] == []
    assert result["raw_extraction"] == []


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".extract_file_sync")
def test_edge_sequential_extraction_error_skipped(
    mock_extract: MagicMock, sequential_converter: KreuzbergConverter
) -> None:
    mock_extract.side_effect = RuntimeError("extraction failed")
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    assert result["documents"] == []
    assert result["raw_extraction"] == []


def _make_mock_result(**overrides: Any) -> MagicMock:
    """Create a mock ExtractionResult with all attributes defaulting to None."""
    result = MagicMock(spec=ExtractionResult)
    defaults: dict[str, Any] = {
        "content": "",
        "metadata": None,
        "quality_score": None,
        "processing_warnings": None,
        "detected_languages": None,
        "extracted_keywords": None,
        "output_format": None,
        "result_format": None,
        "mime_type": None,
        "tables": None,
        "images": None,
        "annotations": None,
        "pages": None,
        "chunks": None,
    }
    for key, default in defaults.items():
        setattr(result, key, overrides.get(key, default))
    return result


@pytest.mark.unit
def test_metadata_mock_processing_warnings() -> None:
    warning = MagicMock()
    warning.source = "ocr"
    warning.message = "low confidence"
    result = _make_mock_result(processing_warnings=[warning])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["processing_warnings"] == [{"source": "ocr", "message": "low confidence"}]


@pytest.mark.unit
def test_metadata_mock_images_excludes_binary_data() -> None:
    result = _make_mock_result(
        images=[
            {
                "format": "png",
                "page_number": 1,
                "width": 200,
                "height": 100,
                "description": "chart",
                "image_index": 0,
                "data": b"binary_data_here",
            },
        ]
    )

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["image_count"] == 1
    assert meta["images"][0]["format"] == "png"
    assert meta["images"][0]["width"] == 200
    assert meta["images"][0]["height"] == 100
    assert meta["images"][0]["description"] == "chart"
    assert meta["images"][0]["image_index"] == 0
    assert meta["images"][0]["page_number"] == 1


@pytest.mark.unit
def test_metadata_mock_annotations() -> None:
    ann = MagicMock()
    ann.annotation_type = "highlight"
    ann.content = "important text"
    ann.page_number = 3
    result = _make_mock_result(annotations=[ann])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["annotations"] == [
        {"type": "highlight", "content": "important text", "page_number": 3},
    ]


@pytest.mark.unit
def test_metadata_mock_tables() -> None:
    table = MagicMock()
    table.cells = [["A", "B"], ["1", "2"]]
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    table.page_number = 1
    result = _make_mock_result(tables=[table])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["table_count"] == 1
    assert meta["tables"][0]["markdown"] == "| A | B |\n|---|---|\n| 1 | 2 |"
    assert meta["tables"][0]["page_number"] == 1


@pytest.mark.unit
def test_metadata_mock_all_fields_populated() -> None:
    warning = MagicMock()
    warning.source = "parser"
    warning.message = "skipped element"
    ann = MagicMock()
    ann.annotation_type = "link"
    ann.content = "https://example.com"
    ann.page_number = 1
    result = _make_mock_result(
        processing_warnings=[warning],
        images=[
            {
                "format": "jpeg",
                "page_number": 2,
                "width": 640,
                "height": 480,
                "description": None,
                "image_index": 0,
                "data": b"...",
            }
        ],
        annotations=[ann],
        quality_score=0.95,
        detected_languages=["en"],
        output_format="markdown",
        result_format="unified",
        mime_type="application/pdf",
    )

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["quality_score"] == 0.95
    assert meta["detected_languages"] == ["en"]
    assert meta["output_format"] == "markdown"
    assert meta["result_format"] == "unified"
    assert meta["mime_type"] == "application/pdf"
    assert meta["processing_warnings"][0]["source"] == "parser"
    assert meta["image_count"] == 1
    assert meta["annotations"][0]["type"] == "link"


@pytest.mark.unit
def test_metadata_file_extensions_mock() -> None:
    result = _make_mock_result(mime_type="application/pdf")
    converter = KreuzbergConverter()

    with patch(f"{CONVERTER_MODULE}.get_extensions_for_mime", return_value=["pdf"]):
        meta = converter._build_extraction_metadata(result)

    assert meta["mime_type"] == "application/pdf"
    assert meta["file_extensions"] == ["pdf"]


@pytest.mark.unit
def test_metadata_no_file_extensions_when_no_mime() -> None:
    result = _make_mock_result(mime_type=None)
    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)

    assert "file_extensions" not in meta


@pytest.mark.unit
def test_chunked_creates_one_document_per_chunk() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    chunk1 = MagicMock()
    chunk1.content = "chunk one"
    chunk1.embedding = [0.1, 0.2, 0.3]
    chunk2 = MagicMock()
    chunk2.content = "chunk two"
    chunk2.embedding = None
    result.chunks = [chunk1, chunk2]

    docs = converter._create_chunked_documents(
        result,
        base_meta={"output_format": "plain"},
        source_meta={"file_path": "test.pdf"},
        user_meta={"custom": "value"},
    )
    assert len(docs) == 2

    assert docs[0].content == "chunk one"
    assert docs[0].embedding == [0.1, 0.2, 0.3]
    assert docs[0].meta["chunk_index"] == 0
    assert docs[0].meta["total_chunks"] == 2
    assert docs[0].meta["output_format"] == "plain"
    assert docs[0].meta["file_path"] == "test.pdf"
    assert docs[0].meta["custom"] == "value"

    assert docs[1].content == "chunk two"
    assert docs[1].embedding is None
    assert docs[1].meta["chunk_index"] == 1
    assert docs[1].meta["total_chunks"] == 2


@pytest.mark.unit
def test_chunked_single_chunk() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    chunk = MagicMock()
    chunk.content = "only chunk"
    chunk.embedding = None
    result.chunks = [chunk]

    docs = converter._create_chunked_documents(
        result,
        base_meta={},
        source_meta={"file_path": "doc.txt"},
        user_meta={},
    )
    assert len(docs) == 1
    assert docs[0].meta["chunk_index"] == 0
    assert docs[0].meta["total_chunks"] == 1


@pytest.mark.unit
def test_per_page_mock_with_object_tables() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    result.output_format = "plain"
    table = MagicMock()
    table.markdown = "| X |\n|---|\n| 1 |"
    table.cells = [["X"], ["1"]]
    table.page_number = 1
    result.pages = [
        {
            "page_number": 1,
            "content": "Page one text",
            "is_blank": False,
            "tables": [table],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={"mime_type": "application/pdf"},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert len(docs) == 1
    assert "| X |" in docs[0].content
    assert "Page one text" in docs[0].content
    assert docs[0].meta["table_count"] == 1
    assert docs[0].meta["page_number"] == 1


@pytest.mark.unit
def test_per_page_mock_with_dict_tables() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    result.output_format = "plain"
    result.pages = [
        {
            "page_number": 1,
            "content": "Page text",
            "is_blank": False,
            "tables": [
                {"markdown": "| Y |\n|---|\n| 2 |", "cells": [["Y"], ["2"]], "page_number": 1},
            ],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert "| Y |" in docs[0].content
    assert docs[0].meta["table_count"] == 1
    assert docs[0].meta["tables"][0]["cells"] == [["Y"], ["2"]]


@pytest.mark.unit
def test_per_page_mock_with_images() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    result.pages = [
        {
            "page_number": 1,
            "content": "Page with image",
            "is_blank": False,
            "tables": [],
            "images": [
                {"format": "jpeg", "page_number": 1, "width": 640, "height": 480, "data": b"img_bytes"},
            ],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert docs[0].meta["image_count"] == 1
    images = docs[0].meta["images"]
    assert images[0]["format"] == "jpeg"
    assert images[0]["width"] == 640
    assert "data" not in images[0]


@pytest.mark.unit
def test_per_page_mock_without_tables_removes_document_level_table_meta() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    result.pages = [
        {
            "page_number": 1,
            "content": "No tables here",
            "is_blank": False,
            "tables": [],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={"table_count": 5, "tables": [{"markdown": "..."}]},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    # Document-level table info should be removed for pages without tables
    assert "table_count" not in docs[0].meta
    assert "tables" not in docs[0].meta


@pytest.mark.unit
def test_deepcopy_per_page_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    result.pages = [
        {"page_number": 1, "content": "Page 1", "is_blank": False, "tables": [], "images": []},
        {"page_number": 2, "content": "Page 2", "is_blank": False, "tables": [], "images": []},
    ]

    user_meta = {"tags": ["original"]}
    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta=user_meta,
    )
    assert len(docs) == 2

    # Mutate one document's nested meta
    docs[0].meta["tags"].append("mutated")

    # Other document and original must be unaffected
    assert docs[1].meta["tags"] == ["original"]
    assert user_meta["tags"] == ["original"]


@pytest.mark.unit
def test_deepcopy_chunked_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
    result = MagicMock()
    chunk1 = MagicMock()
    chunk1.content = "chunk one"
    chunk1.embedding = None
    chunk2 = MagicMock()
    chunk2.content = "chunk two"
    chunk2.embedding = None
    result.chunks = [chunk1, chunk2]

    user_meta = {"tags": ["original"]}
    docs = converter._create_chunked_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta=user_meta,
    )
    assert len(docs) == 2

    docs[0].meta["tags"].append("mutated")

    assert docs[1].meta["tags"] == ["original"]
    assert user_meta["tags"] == ["original"]


@pytest.mark.unit
def test_deepcopy_unified_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
    result = _make_mock_result(content="hello")

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    user_meta = {"tags": ["original"]}
    docs = converter._create_documents(result, bytestream, user_meta)
    assert len(docs) == 1

    docs[0].meta["tags"].append("mutated")

    assert user_meta["tags"] == ["original"]


@pytest.mark.unit
def test_helper_serialize_page_tables_with_dicts() -> None:
    tables = [{"cells": [["A"], ["1"]], "markdown": "| A |", "page_number": 1}]
    result = _serialize_page_tables(tables)
    assert result == [{"cells": [["A"], ["1"]], "markdown": "| A |", "page_number": 1}]


@pytest.mark.unit
def test_helper_serialize_page_tables_with_objects() -> None:
    table = MagicMock()
    table.cells = [["B"], ["2"]]
    table.markdown = "| B |"
    table.page_number = 2
    result = _serialize_page_tables([table])
    assert result == [{"cells": [["B"], ["2"]], "markdown": "| B |", "page_number": 2}]


@pytest.mark.unit
def test_helper_serialize_warnings_with_dicts() -> None:
    warnings = [{"source": "ocr", "message": "low confidence"}]
    result = _serialize_warnings(warnings)
    assert result == [{"source": "ocr", "message": "low confidence"}]


@pytest.mark.unit
def test_helper_serialize_warnings_with_objects() -> None:
    w = MagicMock()
    w.source = "parser"
    w.message = "skipped element"
    result = _serialize_warnings([w])
    assert result == [{"source": "parser", "message": "skipped element"}]
