
import hashlib
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk
from app.services.chunking_service import split_text
from app.services.document_service import extract_text
from app.services.vector_service import VectorService


router = APIRouter(prefix="/documents", tags=["Documents"])
security = HTTPBearer(auto_error=False)
vector_service = VectorService()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


async def verify_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> bool:
    expected_token = settings.nexusai_api_token

    if (
        credentials is None
        or not secrets.compare_digest(
            credentials.credentials,
            expected_token,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def rebuild_vector_index(db: Session) -> None:
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


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_token),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and DOCX files are supported.",
        )

    try:
        # Read at most 10 MB plus one byte to enforce the size limit.
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 10 MB.",
            )

        # Check whether the filename already exists.
        existing_document = (
            db.query(Document)
            .filter(Document.filename == original_filename)
            .first()
        )

        if existing_document:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A document with this filename already exists. "
                    "Please use a different filename."
                ),
            )

        # Calculate SHA-256 to identify identical file contents.
        file_hash = hashlib.sha256(content).hexdigest()

        duplicate_content = (
            db.query(Document)
            .filter(Document.file_hash == file_hash)
            .first()
        )

        if duplicate_content:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This file's content has already been uploaded "
                    f"as '{duplicate_content.filename}'."
                ),
            )

        # Save the file only after both duplicate checks pass.
        safe_filename = f"{uuid4().hex}{extension}"
        file_path = UPLOAD_DIR / safe_filename
        file_path.write_bytes(content)

        try:
            extracted_text = extract_text(file_path)
            chunks = split_text(extracted_text)

            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No usable text was found in the document.",
                )

            document = Document(
                filename=original_filename,
                file_type=extension.lstrip("."),
                file_path=str(file_path),
                file_hash=file_hash,
                status="processed",
            )

            db.add(document)
            db.flush()

            for chunk_index, chunk_content in enumerate(chunks):
                db.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk_index,
                        content=chunk_content,
                    )
                )

            db.commit()
            db.refresh(document)

        except Exception:
            db.rollback()
            file_path.unlink(missing_ok=True)
            raise

        # Rebuild the vector index after the database transaction succeeds.
        try:
            rebuild_vector_index(db)
        except Exception:
            # The document is already saved. Report the indexing failure
            # rather than claiming that the entire upload failed.
            import logging

            logging.getLogger("nexusai.documents").exception(
                "Document saved, but FAISS index rebuilding failed."
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "The document was saved, but indexing failed. "
                    "Please retry indexing before searching this document."
                ),
            )

        return {
            "success": True,
            "document_id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "characters_extracted": len(extracted_text),
            "chunks_created": len(chunks),
            "message": (
                "Document uploaded, saved to PostgreSQL, "
                "and indexed in FAISS."
            ),
        }

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save the document to the database.",
        ) from exc

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded document.",
        ) from exc

    finally:
        db.close()
        await file.close()
