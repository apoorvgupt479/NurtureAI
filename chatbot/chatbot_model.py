"""
=============================================================================
MEDICAL CHATBOT (RAG) MODEL - INPUT FEATURE DOCUMENTATION
=============================================================================

This file contains the core load/predict logic for the Medical Q&A Chatbot.
The model is a Retrieval-Augmented Generation (RAG) system powered by:
  - ChromaDB vector store (embedded via all-MiniLM-L6-v2 ONNX model)
  - Google Gemini (gemma-4-31b-it) for final response generation

The pickle file (chroma_full_state.pkl) stores the full ChromaDB archive
containing 16,407 medical question-answer pairs from the MedQuad dataset.

=============================================================================
INPUT FEATURES REQUIRED (passed to predict() as a dictionary):
=============================================================================

1. query  [REQUIRED]
   - Meaning  : The medical question or symptom query from the user/parent.
   - Data Type: str
   - Example  : "What are the symptoms of whooping cough?"
   - Notes    : Must be a non-empty string. This is the primary search input.

2. child_info  [OPTIONAL]
   - Meaning  : Dictionary containing information about the child patient.
   - Data Type: dict
   - Possible Keys:
       "age"      → Age of the child (str or int). Example: "5" or 5
       "symptoms" → Brief description of symptoms (str). Example: "Mild fever and cough"
   - Example  : {"age": "5", "symptoms": "Mild fever and cough"}
   - Default  : {} (empty dict if not provided)

3. parent_info  [OPTIONAL]
   - Meaning  : Dictionary containing the parent's observation or notes.
   - Data Type: dict
   - Possible Keys:
       "observation" → Parent's own description of the child's condition (str).
                       Example: "Seems lethargic but is drinking water"
   - Example  : {"observation": "Seems lethargic but is drinking water"}
   - Default  : {} (empty dict if not provided)

4. chat_history  [OPTIONAL]
   - Meaning  : List of previous conversation turns for multi-turn context.
   - Data Type: list of dicts, each with keys "user" (str) and "assistant" (str)
   - Example  : [{"user": "What is fever?", "assistant": "Fever is ..."}]
   - Default  : [] (empty list for a fresh conversation)

5. n_results  [OPTIONAL]
   - Meaning  : Number of top RAG documents to retrieve from the vector store.
   - Data Type: int
   - Valid Range: 1 to 20 (recommended: 3–5)
   - Default  : 3

6. google_api_key  [REQUIRED]
   - Meaning  : Your Google Gemini API key for response generation.
   - Data Type: str
   - Valid Values: A valid Google AI Studio / Vertex AI API key string.
   - Notes    : Can also be set via the GOOGLE_API_KEY environment variable.
                If provided in input_data, it overrides the environment variable.

=============================================================================
OUTPUT FORMAT:
=============================================================================
predict() returns a dict:
  {
      "status": "success" | "error",
      "code": 200 | 400 | 500,
      "message": "<Gemini-generated medical response>",
      "isSerious": true | false,
      "shouldCallEmergency": true | false,
      "rag_documents": ["<matched question 1>", ...],   # top retrieved docs
      "sources": ["<source 1>", ...]                    # source names
  }

=============================================================================
EXAMPLE USAGE:
=============================================================================
  load()
  result = predict({
      "query": "My child has a rash and fever",
      "child_info": {"age": "3", "symptoms": "Rash and high fever"},
      "parent_info": {"observation": "Rash appeared this morning"},
      "chat_history": [],
      "n_results": 3,
      "google_api_key": "YOUR_API_KEY_HERE"
  })
  print(result)

=============================================================================
"""

import numpy as np
if not hasattr(np, 'iterable'):
    np.iterable = lambda x: hasattr(x, '__iter__')

import os
import pickle
import json
import copy

# Global state
_chroma_collection = None
_chroma_client = None
_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_full_state.pkl")

# ---------------------------------------------------------------------------
# LOAD FUNCTION
# ---------------------------------------------------------------------------
def load(pkl_path: str = None) -> dict:
    """
    Loads the chroma_full_state.pkl archive into an in-memory ChromaDB
    collection so it is ready for querying.

    Args:
        pkl_path (str, optional): Absolute or relative path to the .pkl file.
                                  Defaults to chroma_full_state.pkl in the
                                  same directory as this script.

    Returns:
        dict: {"status": "success", "code": 200, "message": "..."}
              or {"status": "error",   "code": 500, "message": "<error>"}
    """
    global _chroma_collection, _chroma_client, _pkl_path

    try:
        import chromadb
    except ImportError:
        return {
            "status": "error",
            "code": 500,
            "message": "chromadb is not installed. Run: pip install chromadb"
        }

    # Resolve path
    if pkl_path:
        _pkl_path = pkl_path

    if not os.path.exists(_pkl_path):
        return {
            "status": "error",
            "code": 500,
            "message": f"PKL file not found at: {_pkl_path}"
        }

    try:
        # Load archive
        with open(_pkl_path, "rb") as f:
            archive = pickle.load(f)

        # Validate archive keys
        required_keys = {"ids", "embeddings", "documents", "metadatas"}
        missing = required_keys - set(archive.keys())
        if missing:
            return {
                "status": "error",
                "code": 500,
                "message": f"PKL archive is missing keys: {missing}"
            }

        # Build in-memory ChromaDB collection
        _chroma_client = chromadb.Client()
        col_name = "medquad_chatbot"

        # Clear any previous collection
        try:
            _chroma_client.delete_collection(col_name)
        except Exception:
            pass

        _chroma_collection = _chroma_client.create_collection(name=col_name)

        # Insert in batches to avoid memory/API limits
        total = len(archive["ids"])
        batch_size = 5000
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            _chroma_collection.upsert(
                ids=archive["ids"][i:end],
                embeddings=archive["embeddings"][i:end],
                documents=archive["documents"][i:end],
                metadatas=archive["metadatas"][i:end]
            )

        loaded_count = _chroma_collection.count()
        return {
            "status": "success",
            "code": 200,
            "message": f"Model loaded successfully. ChromaDB collection ready with {loaded_count} items."
        }

    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": str(e)
        }


# ---------------------------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------------------------
def predict(input_data: dict) -> dict:
    """
    Takes user input, retrieves relevant medical documents via ChromaDB RAG,
    and generates a structured medical response using Google Gemini.

    Args:
        input_data (dict): Must contain at minimum:
            - "query"          (str)  : The medical question.
            - "google_api_key" (str)  : Gemini API key (or set GOOGLE_API_KEY env var).
          Optionally:
            - "child_info"     (dict) : Child age and symptoms.
            - "parent_info"    (dict) : Parent observations.
            - "chat_history"   (list) : Previous conversation turns.
