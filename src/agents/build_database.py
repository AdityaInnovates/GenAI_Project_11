"""
==============================================================================
 build_database.py — LIAR Dataset Ingestion into ChromaDB
==============================================================================

 PURPOSE:
   This is a one-time (or re-runnable) script that:
     1. Downloads the LIAR fact-checking dataset from HuggingFace.
     2. Converts each record into a LangChain Document with structured text.
     3. Embeds the documents using a local sentence-transformer model.
     4. Persists everything into a local ChromaDB vector store at ./chroma_db.

 USAGE:
   cd intelligent-news-credibility/Agent
   python build_database.py

 NOTES:
   - We only load the first 1000 records (split="train[:1000]") for fast
     local testing. Increase this for production use.
   - The embedding model (all-MiniLM-L6-v2) runs locally — no API keys needed.
   - Re-running this script will rebuild the database from scratch.
==============================================================================
"""

import os
import shutil
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==============================================================================
# STEP 1: Configuration & Constants
# ==============================================================================

# --- Label Mapping ---
# The LIAR dataset encodes verdicts as integers (0–5).
# This dictionary maps them to human-readable truthfulness labels.
LABEL_MAP = {
    0: "Pants on Fire (Completely False)",
    1: "False",
    2: "Barely True",
    3: "Half True",
    4: "Mostly True",
    5: "True",
}

# --- Database Path ---
# The Chroma database will be persisted to this local directory.
# Both this script and the LangGraph agent will reference the same path.
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chroma_db")

# --- Embedding Model ---
# We use a lightweight, high-quality sentence-transformer that runs on CPU.
# Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Dataset Slice ---
# Only load the first 1000 records from the training split for fast testing.
DATASET_SPLIT = "train[:1000]"


# ==============================================================================
# STEP 2: Load the LIAR Dataset from HuggingFace
# ==============================================================================

def load_liar_dataset(split: str = DATASET_SPLIT):
    """
    Downloads (or loads from cache) the LIAR dataset from HuggingFace Hub.

    The LIAR dataset contains ~12K human-labeled short political statements
    with 6-class truthfulness ratings from PolitiFact.

    Args:
        split: Which slice to load. Default is "train[:1000]" for testing.

    Returns:
        A HuggingFace Dataset object.
    """
    print(f"📥 Loading LIAR dataset (split: {split})...")
    dataset = load_dataset("liar", split=split, trust_remote_code=True)
    print(f"   ✅ Loaded {len(dataset)} records.")
    return dataset


# ==============================================================================
# STEP 3: Convert Dataset Rows → LangChain Documents
# ==============================================================================

def format_documents(dataset) -> list[Document]:
    """
    Iterates through every row of the LIAR dataset and creates a
    LangChain Document for each one.

    Each Document has:
      - page_content: A clean, readable string combining:
            • The factual statement
            • Who said it (speaker)
            • The context/venue where it was said
            • The official PolitiFact verdict
      - metadata: A dictionary containing the verdict string
            (useful for filtering and display later).

    Args:
        dataset: The HuggingFace LIAR dataset object.

    Returns:
        A list of LangChain Document objects ready for embedding.
    """
    documents = []
    skipped = 0  # Counter for rows we couldn't process

    print("📝 Formatting dataset rows into LangChain Documents...")

    for i, row in enumerate(dataset):
        # --- Extract fields from the row ---
        statement = row.get("statement", "").strip()
        speaker = row.get("speaker", "Unknown").strip()
        context = row.get("context", "Unknown context").strip()
        label_id = row.get("label", -1)

        # --- Skip rows with missing statements ---
        if not statement:
            skipped += 1
            continue

        # --- Map the integer label to a readable verdict string ---
        verdict = LABEL_MAP.get(label_id, "Unknown Verdict")

        # --- Build a clean, structured page_content string ---
        # This is what gets embedded and searched against later.
        page_content = (
            f"Statement: \"{statement}\"\n"
            f"Speaker: {speaker}\n"
            f"Context: {context}\n"
            f"Verdict: {verdict}"
        )

        # --- Create the LangChain Document ---
        doc = Document(
            page_content=page_content,
            metadata={
                "verdict": verdict,        # Store verdict for easy access
                "speaker": speaker,         # Useful for filtering by speaker
                "source": "LIAR Dataset",   # Provenance tracking
            },
        )
        documents.append(doc)

    print(f"   ✅ Created {len(documents)} documents. (Skipped {skipped} empty rows)")
    return documents


# ==============================================================================
# STEP 4: Initialize the Embedding Model
# ==============================================================================

def get_embedding_function():
    """
    Initializes and returns a HuggingFaceEmbeddings instance using the
    all-MiniLM-L6-v2 sentence-transformer model.

    This model:
      - Runs entirely locally (no API key needed)
      - Produces 384-dimensional embeddings
      - Is optimized for semantic similarity tasks
      - Has a good balance of speed and quality

    Returns:
        A HuggingFaceEmbeddings object.
    """
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # Force CPU (safe for all machines)
        encode_kwargs={"normalize_embeddings": True},  # L2-normalize for cosine sim
    )
    print("   ✅ Embedding model loaded.")
    return embeddings


# ==============================================================================
# STEP 5: Build & Persist the ChromaDB Vector Store
# ==============================================================================

def build_chroma_db(documents: list[Document], embeddings):
    """
    Takes the list of LangChain Documents and the embedding function,
    embeds all documents, and persists them to a local ChromaDB directory.

    If the chroma_db directory already exists, it is deleted and rebuilt
    from scratch to ensure a clean state.

    Args:
        documents: List of LangChain Documents to embed.
        embeddings: The HuggingFaceEmbeddings instance.
    """
    # --- Clean up any existing database ---
    if os.path.exists(CHROMA_DB_DIR):
        print(f"🗑️  Removing existing database at: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)

    print(f"🔨 Building ChromaDB vector store at: {CHROMA_DB_DIR}")
    print(f"   Embedding {len(documents)} documents... (this may take a minute)")

    # --- Create the Chroma vector store ---
    # Chroma.from_documents() does three things:
    #   1. Embeds each document's page_content using the embedding function
    #   2. Stores the embeddings + metadata in the vector store
    #   3. Persists everything to the specified directory
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="liar_fact_checks",  # Name of the collection inside Chroma
    )

    print(f"   ✅ ChromaDB built and persisted successfully!")
    print(f"   📁 Database location: {CHROMA_DB_DIR}")
    print(f"   📊 Total vectors stored: {vectorstore._collection.count()}")

    return vectorstore


# ==============================================================================
# STEP 6: Main Entry Point
# ==============================================================================

def main():
    """
    Orchestrates the full ingestion pipeline:
      1. Load the LIAR dataset
      2. Format rows into Documents
      3. Initialize embeddings
      4. Build & persist the ChromaDB
    """
    print("=" * 60)
    print(" 🚀 LIAR Dataset → ChromaDB Ingestion Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load dataset
    dataset = load_liar_dataset()

    # Step 2: Format into LangChain Documents
    documents = format_documents(dataset)

    # Step 3: Initialize embedding model
    embeddings = get_embedding_function()

    # Step 4: Build and persist the vector database
    build_chroma_db(documents, embeddings)

    print()
    print("=" * 60)
    print(" ✅ INGESTION COMPLETE — Database is ready for the agent!")
    print("=" * 60)


# --- Run the pipeline when executed directly ---
if __name__ == "__main__":
    main()
