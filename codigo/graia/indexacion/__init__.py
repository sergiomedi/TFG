"""Subsistema de indexación de GRAIA.

Componentes principales:
  - :class:`Embedder`    — genera vectores densos con E5 multilingüe
  - :class:`VectorStore` — almacén FAISS IndexFlatIP con persistencia

Flujo típico::

    embedder = Embedder()
    vectors = embedder.encode_passages([c.text for c in chunks])
    store = VectorStore(dim=embedder.dim)
    store.add(vectors, chunks)
    store.save("data/index")
"""

from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore

__all__ = ["Embedder", "VectorStore"]
