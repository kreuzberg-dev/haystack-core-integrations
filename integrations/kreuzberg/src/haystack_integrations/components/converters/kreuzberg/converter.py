# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import tempfile
from pathlib import Path
from typing import Any, cast

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream

from kreuzberg import (
    ExtractionConfig,
    ExtractionResult,
    LanguageDetectionConfig,
    batch_extract_bytes_sync,
    batch_extract_files_sync,
    config_merge,
    config_to_json,
    detect_mime_type,
    error_code_name,
    extract_bytes_sync,
    extract_file_sync,
    get_error_details,
    get_last_error_code,
    list_document_extractors,
    list_ocr_backends,
    load_extraction_config_from_file,
)

logger = logging.getLogger(__name__)

# Metadata keys that duplicate top-level ExtractionResult fields.
# These are excluded when flattening result.metadata into Document.meta
# because we handle them from the top-level fields instead.
_METADATA_OVERLAP_KEYS = frozenset({"quality_score", "output_format", "keywords"})


@component
class KreuzbergConverter:
    """
    Converts files to Documents using [Kreuzberg](https://docs.kreuzberg.dev/).

    Kreuzberg is a document intelligence framework that extracts text from
    PDFs, Office documents, images, and 75+ other formats. All processing
    is performed locally with no external API calls.

    **Usage Example:**

    ```python
    from haystack_integrations.components.converters.kreuzberg import (
        KreuzbergConverter,
    )

    converter = KreuzbergConverter()
    result = converter.run(sources=["document.pdf", "report.docx"])
    documents = result["documents"]
    ```

    You can also pass kreuzberg's `ExtractionConfig` to customize extraction:

    ```python
    from kreuzberg import ExtractionConfig, OcrConfig

    converter = KreuzbergConverter(
        config=ExtractionConfig(
            output_format="markdown",
            ocr=OcrConfig(backend="tesseract", language="eng"),
        ),
    )
    ```

    The converter exposes two output sockets: `documents` and
    `raw_extraction`. The `raw_extraction` output contains the serialized
    kreuzberg `ExtractionResult` for each source, useful for debugging or
    advanced downstream processing.
    """

    def __init__(
        self,
        *,
        config: ExtractionConfig | None = None,
        config_path: str | Path | None = None,
        store_full_path: bool = False,
        batch: bool = True,
        append_tables_to_content: bool = True,
        easyocr_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a `KreuzbergConverter` component.

        :param config:
            An optional `kreuzberg.ExtractionConfig` object to customize
            extraction behavior. Use this to set output format, OCR backend
            and language, force-OCR mode, per-page extraction, chunking,
            keyword extraction, and other kreuzberg options. If not provided,
            kreuzberg's defaults are used.
        :param config_path:
            Path to a kreuzberg configuration file (`.toml`, `.yaml`, or
            `.json`). When both `config` and `config_path` are provided,
            `config` takes precedence (is merged on top of the file config).
        :param store_full_path:
            If `True`, the full file path is stored in the Document metadata.
            If `False`, only the file name is stored.
        :param batch:
            If `True`, use kreuzberg's batch extraction APIs which leverage
            Rust's rayon thread pool for parallel processing. If `False`,
            sources are extracted one at a time.
        :param append_tables_to_content:
            If `True`, append extracted table markdown to the end of each
            Document's content.
        :param easyocr_kwargs:
            Optional keyword arguments to pass to EasyOCR when using the
            `"easyocr"` backend. Supports GPU, beam width, model storage,
            and other EasyOCR-specific options.
        """
        self.config = config
        self.config_path = str(config_path) if config_path is not None else None
        self.store_full_path = store_full_path
        self.batch = batch
        self.append_tables_to_content = append_tables_to_content
        self.easyocr_kwargs = easyocr_kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        config_json = config_to_json(self.config) if self.config else None
        return default_to_dict(
            self,
            config=config_json,
            config_path=self.config_path,
            store_full_path=self.store_full_path,
            batch=self.batch,
            append_tables_to_content=self.append_tables_to_content,
            easyocr_kwargs=self.easyocr_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KreuzbergConverter":
        """
        Deserialize this component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        config_data = init_params.get("config")
        if isinstance(config_data, str):
            # JSON string from config_to_json — round-trip via temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(config_data)
                tmp_path = f.name
            try:
                init_params["config"] = load_extraction_config_from_file(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _build_config(self) -> ExtractionConfig:
        """
        Build the effective `ExtractionConfig` by merging all configuration
        sources in priority order: file config < explicit config.

        Always returns a *fresh* config object — never mutates ``self.config``.
        """
        # Determine base config (never mutate self.config)
        if self.config is not None:
            # Deep copy via JSON round-trip to avoid mutating the user's object
            config = _copy_config(self.config)
        elif self.config_path is not None:
            config = load_extraction_config_from_file(self.config_path)
        else:
            config = ExtractionConfig()

        # When both config and config_path are provided, fill any None gaps
        # in config from the file config (config takes priority).
        if self.config is not None and self.config_path is not None:
            file_config = load_extraction_config_from_file(self.config_path)
            config_merge(config, file_config)

        # Auto-enable language detection if not explicitly configured
        if config.language_detection is None:
            config.language_detection = LanguageDetectionConfig(enabled=True)

        return config

    # ------------------------------------------------------------------
    # Source handling
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_sources(sources: list[str | Path | ByteStream]) -> list[str | Path | ByteStream]:
        """
        Expand directory paths in the sources list to their direct children.

        Directories are expanded non-recursively (only direct file children).
        Files are sorted alphabetically for deterministic ordering.
        """
        expanded: list[str | Path | ByteStream] = []
        for source in sources:
            if isinstance(source, ByteStream):
                expanded.append(source)
            else:
                path = Path(source)
                if path.is_dir():
                    expanded.extend(sorted(path.glob("*.*")))
                else:
                    expanded.append(source)
        return expanded

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_single(
        self,
        source: str | Path | ByteStream,
        config: ExtractionConfig,
    ) -> ExtractionResult:
        """
        Extract content from a single source using kreuzberg.
        """
        if isinstance(source, ByteStream):
            mime_type = source.mime_type or detect_mime_type(source.data)
            return extract_bytes_sync(
                source.data,
                mime_type=mime_type,
                config=config,
                easyocr_kwargs=self.easyocr_kwargs,
            )
        return extract_file_sync(source, config=config, easyocr_kwargs=self.easyocr_kwargs)

    def _extract_batch(
        self,
        sources: list[str | Path | ByteStream],
        config: ExtractionConfig,
    ) -> list[ExtractionResult | None]:
        """
        Extract content from multiple sources using kreuzberg's batch APIs.

        Returns ``None`` for sources that were skipped (e.g. unreadable).
        Exceptions from batch extraction are propagated to the caller.
        """
        file_indices: list[int] = []
        file_paths: list[str | Path] = []
        bytes_indices: list[int] = []
        bytes_data: list[bytes | bytearray] = []
        bytes_mimes: list[str] = []

        for i, source in enumerate(sources):
            if isinstance(source, ByteStream):
                bytes_indices.append(i)
                bytes_data.append(source.data)
                bytes_mimes.append(source.mime_type or detect_mime_type(source.data))
            else:
                file_indices.append(i)
                file_paths.append(source)

        results: list[ExtractionResult | None] = [None] * len(sources)

        # Batch-extract file paths
        if file_paths:
            file_results = batch_extract_files_sync(file_paths, config=config, easyocr_kwargs=self.easyocr_kwargs)
            for idx, result in zip(file_indices, file_results, strict=True):
                results[idx] = result

        # Batch-extract byte streams
        if bytes_data:
            bytes_results = batch_extract_bytes_sync(
                bytes_data, bytes_mimes, config=config, easyocr_kwargs=self.easyocr_kwargs
            )
            for idx, result in zip(bytes_indices, bytes_results, strict=True):
                results[idx] = result

        return results

    # ------------------------------------------------------------------
    # Metadata & content assembly
    # ------------------------------------------------------------------

    def _build_extraction_metadata(self, result: ExtractionResult) -> dict[str, Any]:
        """
        Build metadata dict from an ``ExtractionResult``, flattening kreuzberg's
        metadata fields and enriching with top-level result attributes.

        None values are filtered out.
        """
        # Flatten kreuzberg document metadata (format-specific TypedDict)
        meta: dict[str, Any] = (
            {k: v for k, v in result.metadata.items() if v is not None and k not in _METADATA_OVERLAP_KEYS}
            if result.metadata
            else {}
        )

        # Quality score
        if result.quality_score is not None:
            meta["quality_score"] = result.quality_score

        # Processing warnings
        if result.processing_warnings:
            meta["processing_warnings"] = _serialize_warnings(result.processing_warnings)

        # Detected languages
        if result.detected_languages:
            meta["detected_languages"] = list(result.detected_languages)

        # Extracted keywords
        if result.extracted_keywords:
            meta["extracted_keywords"] = [
                {"text": kw.text, "score": kw.score, "algorithm": kw.algorithm} for kw in result.extracted_keywords
            ]

        # Output format tracking
        if result.output_format:
            meta["output_format"] = str(result.output_format)
        if result.result_format:
            meta["result_format"] = str(result.result_format)
        if result.mime_type:
            meta["mime_type"] = result.mime_type

        # Tables metadata
        if result.tables:
            meta["table_count"] = len(result.tables)
            meta["tables"] = _serialize_tables(result.tables)

        # Image extraction metadata (no binary data)
        if result.images:
            meta["image_count"] = len(result.images)
            meta["images"] = [
                {
                    "format": img.get("format"),
                    "page_number": img.get("page_number"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "description": img.get("description"),
                    "image_index": img.get("image_index"),
                }
                for img in result.images
            ]

        # PDF annotations
        if result.annotations:
            meta["annotations"] = _serialize_annotations(result.annotations)

        return meta

    def _assemble_content(self, text: str, tables: list[Any] | None) -> str:
        """
        Assemble document content, optionally appending table markdown.
        """
        if not self.append_tables_to_content or not tables:
            return text

        table_blocks = [t.markdown for t in tables if t.markdown]
        if not table_blocks:
            return text

        return text + "\n\n" + "\n\n".join(table_blocks)

    # ------------------------------------------------------------------
    # Document creation
    # ------------------------------------------------------------------

    def _create_documents(
        self,
        result: ExtractionResult,
        bytestream: ByteStream,
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """
        Create one or more ``Document`` objects from an ``ExtractionResult``.

        Output mode depends on what kreuzberg returns:
        - **Default:** one Document per source (unified content).
        - **Pages present:** one Document per page (when
          ``PageConfig(extract_pages=True)`` is in the config).
        - **Chunks present:** one Document per chunk (when
          ``ChunkingConfig`` is in the config).
        """
        base_meta = self._build_extraction_metadata(result)

        # Source metadata from bytestream
        source_meta = dict(bytestream.meta)
        if not self.store_full_path and "file_path" in source_meta:
            source_meta["file_path"] = Path(source_meta["file_path"]).name

        # Chunking mode — one Document per chunk
        if result.chunks:
            return self._create_chunked_documents(result, base_meta, source_meta, user_meta)

        # Per-page mode — one Document per page
        if result.pages:
            return self._create_per_page_documents(result, base_meta, source_meta, user_meta)

        # Default: unified mode — one Document per source
        content = self._assemble_content(result.content, result.tables)
        merged = {**base_meta, **source_meta, **copy.deepcopy(user_meta)}
        return [Document(content=content, meta=merged)]

    def _create_per_page_documents(
        self,
        result: ExtractionResult,
        base_meta: dict[str, Any],
        source_meta: dict[str, Any],
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """Create one Document per page."""
        documents: list[Document] = []
        for page in result.pages:
            page_content = page.get("content", "")
            page_tables = page.get("tables", [])
            page_images = page.get("images", [])

            if self.append_tables_to_content and page_tables:
                table_blocks = [t.markdown for t in page_tables if hasattr(t, "markdown") and t.markdown]
                if not table_blocks:
                    # Tables might be dicts in page context
                    table_blocks = [t["markdown"] for t in page_tables if isinstance(t, dict) and t.get("markdown")]
                if table_blocks:
                    page_content = page_content + "\n\n" + "\n\n".join(table_blocks)

            page_meta: dict[str, Any] = {
                **base_meta,
                **source_meta,
                "page_number": page.get("page_number"),
                "is_blank": page.get("is_blank", False),
            }

            if page_tables:
                page_meta["table_count"] = len(page_tables)
                page_meta["tables"] = _serialize_page_tables(page_tables)

            if page_images:
                page_meta["image_count"] = len(page_images)
                page_meta["images"] = [
                    {k: v for k, v in img.items() if k != "data"} if isinstance(img, dict) else img
                    for img in page_images
                ]

            # Remove document-level table/image info to avoid confusion
            # (page-level info is more specific)
            page_meta.pop("table_count", None) if not page_tables else None
            page_meta.pop("tables", None) if not page_tables else None

            page_meta.update(copy.deepcopy(user_meta))
            documents.append(Document(content=page_content, meta=page_meta))

        return documents

    def _create_chunked_documents(
        self,
        result: ExtractionResult,
        base_meta: dict[str, Any],
        source_meta: dict[str, Any],
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """Create one Document per chunk, with optional embeddings."""
        documents: list[Document] = []
        total_chunks = len(result.chunks)

        for i, chunk in enumerate(result.chunks):
            chunk_meta = {
                **base_meta,
                **source_meta,
                "chunk_index": i,
                "total_chunks": total_chunks,
                **copy.deepcopy(user_meta),
            }
            documents.append(
                Document(
                    content=chunk.content,
                    embedding=chunk.embedding,
                    meta=chunk_meta,
                )
            )

        return documents

    # ------------------------------------------------------------------
    # Raw extraction output
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_result(result: ExtractionResult) -> dict[str, Any]:
        """
        Serialize an ``ExtractionResult`` to a plain dict for the
        ``raw_extraction`` output socket.
        """
        raw: dict[str, Any] = {
            "content": result.content,
            "mime_type": result.mime_type,
            "output_format": str(result.output_format) if result.output_format else None,
            "result_format": str(result.result_format) if result.result_format else None,
        }

        if result.metadata:
            raw["metadata"] = dict(result.metadata)

        if result.tables:
            raw["tables"] = _serialize_tables(result.tables)

        if result.quality_score is not None:
            raw["quality_score"] = result.quality_score

        if result.detected_languages:
            raw["detected_languages"] = list(result.detected_languages)

        if result.processing_warnings:
            raw["processing_warnings"] = _serialize_warnings(result.processing_warnings)

        if result.extracted_keywords:
            raw["extracted_keywords"] = [
                {"text": kw.text, "score": kw.score, "algorithm": kw.algorithm} for kw in result.extracted_keywords
            ]

        if result.annotations:
            raw["annotations"] = _serialize_annotations(result.annotations)

        if result.pages:
            raw["pages"] = [
                {
                    "page_number": p.get("page_number"),
                    "content": p.get("content"),
                    "is_blank": p.get("is_blank"),
                    "tables": _serialize_page_tables(p.get("tables", [])),
                }
                for p in result.pages
            ]

        if result.chunks:
            raw["chunks"] = [{"content": c.content, "metadata": c.metadata} for c in result.chunks]

        if result.images:
            raw["images"] = [{k: v for k, v in img.items() if k != "data"} for img in result.images]

        return raw

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _log_extraction_error(source: Any, error: Exception) -> None:
        """
        Log a structured extraction error using kreuzberg's error
        diagnostics when available.
        """
        try:
            error_code = get_last_error_code()
            details = get_error_details()
            code_name = error_code_name(error_code) if error_code is not None else "UNKNOWN"
            logger.warning(
                "Could not convert {source} to Document. Error code: {code} ({name}). Details: {details}. Skipping it.",
                source=source,
                code=error_code,
                name=code_name,
                details=details.get("message", str(error)),
            )
        except Exception:
            logger.warning(
                "Could not convert {source} to Document. Skipping it. Error: {error}",
                source=source,
                error=error,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document], raw_extraction=list[dict[str, Any]])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document] | list[dict[str, Any]]]:
        """
        Convert files to Documents using Kreuzberg.

        :param sources:
            List of file paths, directory paths, or ByteStream objects to
            convert. Directory paths are expanded to their direct file children
            (non-recursive, sorted alphabetically).
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single
            dictionary. If it's a single dictionary, its content is added to
            the metadata of all produced Documents. If it's a list, the length
            of the list must match the number of sources, because the two
            lists will be zipped. If `sources` contains ByteStream objects,
            their `meta` will be added to the output Documents.

            **Note:** When directories are present in `sources`, `meta` must
            be a single dictionary (not a list), since the number of files in
            a directory is not known in advance.

        :returns:
            A dictionary with the following keys:

            - `documents`: A list of created Documents.
            - `raw_extraction`: A list of serialized kreuzberg
              ExtractionResult dicts, one per successfully processed source.
        """
        # Expand directories
        has_dirs = any(
            isinstance(s, (str, Path)) and not isinstance(s, ByteStream) and Path(s).is_dir() for s in sources
        )
        if has_dirs and isinstance(meta, list):
            msg = (
                "When directories are present in 'sources', 'meta' must be a "
                "single dictionary, not a list, since the number of files in "
                "a directory is not known in advance."
            )
            raise ValueError(msg)

        expanded_sources = self._expand_sources(sources)
        meta_list = normalize_metadata(meta, sources_count=len(expanded_sources))

        config = self._build_config()

        documents: list[Document] = []
        raw_extractions: list[dict[str, Any]] = []

        if self.batch and len(expanded_sources) > 1:
            self._run_batch(expanded_sources, meta_list, config, documents, raw_extractions)
        else:
            self._run_sequential(expanded_sources, meta_list, config, documents, raw_extractions)

        return {"documents": documents, "raw_extraction": raw_extractions}

    def _run_sequential(
        self,
        sources: list[str | Path | ByteStream],
        meta_list: list[dict[str, Any]],
        config: ExtractionConfig,
        documents: list[Document],
        raw_extractions: list[dict[str, Any]],
    ) -> None:
        """Process sources one at a time."""
        for source, user_meta in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            try:
                result = self._extract_single(source, config)
            except Exception as e:
                self._log_extraction_error(source, e)
                continue

            docs = self._create_documents(result, bytestream, user_meta)
            documents.extend(docs)
            raw_extractions.append(self._serialize_result(result))

    def _run_batch(
        self,
        sources: list[str | Path | ByteStream],
        meta_list: list[dict[str, Any]],
        config: ExtractionConfig,
        documents: list[Document],
        raw_extractions: list[dict[str, Any]],
    ) -> None:
        """Process sources using batch extraction."""
        # Pre-validate sources (get bytestreams for metadata)
        bytestreams: list[ByteStream | None] = []
        for source in sources:
            try:
                bytestreams.append(get_bytestream_from_source(source))
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                bytestreams.append(None)

        results = self._extract_batch(sources, config)

        for _source, result, bytestream, user_meta in zip(sources, results, bytestreams, meta_list, strict=True):
            if result is None or bytestream is None:
                continue

            docs = self._create_documents(result, bytestream, user_meta)
            documents.extend(docs)
            raw_extractions.append(self._serialize_result(result))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def supported_extractors() -> list[str]:
        """
        List all document extractors registered in kreuzberg.

        :returns:
            List of extractor names.
        """
        return cast(list[str], list_document_extractors())

    @staticmethod
    def supported_ocr_backends() -> list[str]:
        """
        List all OCR backends registered in kreuzberg.

        :returns:
            List of OCR backend names.
        """
        return cast(list[str], list_ocr_backends())


# ======================================================================
# Module-level serialization helpers
# ======================================================================


def _serialize_tables(tables: list[Any]) -> list[dict[str, Any]]:
    """Serialize ExtractedTable objects to plain dicts."""
    return [
        {
            "cells": t.cells,
            "markdown": t.markdown,
            "page_number": t.page_number,
        }
        for t in tables
    ]


def _serialize_page_tables(tables: list[Any]) -> list[dict[str, Any]]:
    """Serialize tables from a page context (may be objects or dicts)."""
    serialized = []
    for t in tables:
        if isinstance(t, dict):
            serialized.append(
                {
                    "cells": t.get("cells"),
                    "markdown": t.get("markdown"),
                    "page_number": t.get("page_number"),
                }
            )
        else:
            serialized.append(
                {
                    "cells": t.cells,
                    "markdown": t.markdown,
                    "page_number": t.page_number,
                }
            )
    return serialized


def _serialize_warnings(warnings: list[Any]) -> list[dict[str, str]]:
    """Serialize processing warnings to plain dicts."""
    serialized = []
    for w in warnings:
        if isinstance(w, dict):
            serialized.append({"source": w.get("source", ""), "message": w.get("message", "")})
        else:
            serialized.append({"source": getattr(w, "source", ""), "message": getattr(w, "message", "")})
    return serialized


def _serialize_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """Serialize PDF annotations to plain dicts."""
    serialized = []
    for ann in annotations:
        if isinstance(ann, dict):
            serialized.append(dict(ann))
        else:
            serialized.append(
                {
                    "type": getattr(ann, "annotation_type", None),
                    "content": getattr(ann, "content", None),
                    "page_number": getattr(ann, "page_number", None),
                }
            )
    return serialized


def _copy_config(config: ExtractionConfig) -> ExtractionConfig:
    """
    Deep copy an ``ExtractionConfig`` by round-tripping through JSON.

    This is necessary because kreuzberg's PyO3 config objects don't support
    Python's ``copy.deepcopy()`` protocol.
    """
    json_str = config_to_json(config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        tmp_path = f.name
    try:
        return load_extraction_config_from_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
