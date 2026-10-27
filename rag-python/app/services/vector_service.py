
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("nexusai.vector")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MIN_SIMILARITY_SCORE = 0.25


class VectorService:
    def __init__(
        self,
        storage_dir: str | Path = "storage/faiss",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )
        self.chunks: list[dict] = []

    def _embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def add_chunks(
        self,
        document_id: int,
        chunks: list[str],
    ) -> None:
        if not chunks:
            logger.warning(
                "No chunks supplied for document %s.",
                document_id,
            )
            return

        embeddings = self._embed(chunks)

        self.index.add(embeddings)

        for chunk_index, chunk in enumerate(chunks):
            self.chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

        logger.info(
            "Added %s chunks for document %s. "
            "Total indexed chunks: %s",
            len(chunks),
            document_id,
            self.index.ntotal,
        )

    def rebuild(
        self,
        records: list[dict],
    ) -> None:
        """Rebuild the FAISS index from database chunk records."""
        self.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )
        self.chunks = []

        if not records:
            logger.warning(
                "FAISS rebuild received zero database chunk records."
            )
            return

        valid_records = [
            record
            for record in records
            if record.get("content")
            and record["content"].strip()
        ]

        if not valid_records:
            logger.warning(
                "FAISS rebuild found no non-empty chunk content."
            )
            return

        texts = [
            record["content"]
            for record in valid_records
        ]

        embeddings = self._embed(texts)

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{embeddings.shape[1]}"
            )

        self.index.add(embeddings)

        self.chunks = [
            {
                "document_id": record["document_id"],
                "chunk_index": record["chunk_index"],
                "text": record["content"],
            }
            for record in valid_records
        ]

        logger.info(
            "FAISS rebuild complete. Database records: %s; "
            "indexed chunks: %s",
            len(records),
            self.index.ntotal,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Return the highest-scoring unique chunks.

        Similarity filtering is temporarily disabled for diagnostics.
        Review the logged scores before restoring a threshold.
        """
        if not query.strip():
            logger.warning("Search query is empty.")
            return []

        if top_k <= 0:
            return []

        if self.index.ntotal == 0:
            logger.warning(
                "FAISS index is empty. No chunks are indexed."
            )
            return []

        if len(self.chunks) != self.index.ntotal:
            raise RuntimeError(
                "FAISS index and chunk metadata are out of sync."
            )

        query_embedding = self._embed([query])

        scores, indices = self.index.search(
            query_embedding,
            self.index.ntotal,
        )

        logger.info(
            "Searching %s chunks for query: %s",
            self.index.ntotal,
            query,
        )

        results = []
        seen_texts = set()

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):
            if index_id < 0:
                continue

            metadata = self.chunks[int(index_id)]

            normalized_text = " ".join(
                metadata["text"].split()
            ).casefold()

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            results.append(
                {
                    **metadata,
                    "score": float(score),
                }
            )

            if len(results) >= top_k:
                break

        logger.info(
            "Search returned %s unique chunks. Top scores: %s",
            len(results),
            [
                round(item["score"], 4)
                for item in results
            ],
        )

        return results

    def save(self) -> None:
        index_path = self.storage_dir / "index.faiss"
        metadata_path = self.storage_dir / "chunks.json"
        temporary_path = self.storage_dir / "chunks.json.tmp"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        temporary_path.write_text(
            json.dumps(
                self.chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(metadata_path)

        logger.info(
            "Saved FAISS index and %s chunk records.",
            len(self.chunks),
        )

    def load(self) -> None:
        index_path = self.storage_dir / "index.faiss"
        metadata_path = self.storage_dir / "chunks.json"

        if (
            not index_path.exists()
            and not metadata_path.exists()
        ):
            logger.info(
                "No saved FAISS index found."
            )
            return

        if (
            not index_path.exists()
            or not metadata_path.exists()
        ):
            raise RuntimeError(
                "FAISS index or chunk metadata is missing. "
                "Restore both files from a consistent backup."
            )

        loaded_index = faiss.read_index(
            str(index_path)
        )

        loaded_chunks = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if loaded_index.d != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Unexpected FAISS embedding dimension."
            )

        if loaded_index.ntotal != len(loaded_chunks):
            raise RuntimeError(
                "FAISS index and chunk metadata are out of sync."
            )

        self.index = loaded_index
        self.chunks = loaded_chunks

        logger.info(
            "Loaded FAISS index with %s chunks.",
            self.index.ntotal,
        )
