"""
Embedding model setup for the NHL RAG pipeline.

Model:      BAAI/bge-large-en-v1.5
Dimensions: 1024
Device:     CPU (batch_size=16)

BGE models expect a prefix when embedding passages for indexing:
  "Represent this sentence: "
And a different prefix for query-time (used in FastAPI retrieval, not here):
  "Represent this question for searching relevant passages: "
"""
import logging
import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import Chunk

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 16
# BGE passage prefix — prepended to every chunk before embedding
PASSAGE_PREFIX = "Represent this sentence: "


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]   # 1024-dimensional vector


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        log.info(f"Loading embedding model '{model_name}' (first run downloads ~1.3 GB)...")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name, device="cpu")
        elapsed = time.perf_counter() - t0
        log.info(f"Model loaded in {elapsed:.1f}s — embedding dim: {self._model.get_sentence_embedding_dimension()}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunk objects in batches.

        Each chunk's text is prefixed with PASSAGE_PREFIX before encoding,
        as recommended by the BGE authors for document indexing.

        Returns a list of EmbeddedChunk with the original chunk and its vector.
        """
        if not chunks:
            return []

        results: list[EmbeddedChunk] = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch = chunks[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            texts = [PASSAGE_PREFIX + c.text for c in batch]

            t0 = time.perf_counter()
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine similarity works on normalised vectors
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            log.info(
                f"embed batch {batch_idx + 1}/{total_batches}: "
                f"{len(batch)} chunk(s) in {latency_ms:.0f}ms "
                f"({latency_ms / len(batch):.1f}ms/chunk)"
            )

            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=vector.tolist(),
                ))

        return results
