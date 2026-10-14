
from app.db.database import SessionLocal
from app.db.models import DocumentChunk
from app.services.vector_service import VectorService


db = SessionLocal()
vector_service = VectorService()

try:
    records = (
        db.query(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
        )
        .order_by(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        )
        .all()
    )

    vector_service.rebuild(
        [
            {
                "document_id": record.document_id,
                "chunk_index": record.chunk_index,
                "content": record.content,
            }
            for record in records
        ]
    )

    vector_service.save()

    print("PostgreSQL chunks:", len(records))
    print("FAISS vectors:", vector_service.index.ntotal)

    results = vector_service.search(
        "What technologies are used to build NexusAI?",
        top_k=3,
    )

    print("\nSemantic search results:")

    for result in results:
        print("Document ID:", result["document_id"])
        print("Chunk index:", result["chunk_index"])
        print("Similarity score:", round(result["score"], 4))
        print("Text:", result["text"][:300])
        print("-" * 50)

finally:
    db.close()
