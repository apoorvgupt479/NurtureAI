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
_vectorizer = None
_tfidf_matrix = None
_archive = None
_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_documents.pkl")

# ---------------------------------------------------------------------------
# LOAD FUNCTION
# ---------------------------------------------------------------------------
def load(pkl_path: str = None) -> dict:
    """
    Loads the chroma_documents.pkl archive and builds a TF-IDF vector space
    representing the documents for fast medical Q&A retrieval.

    Args:
        pkl_path (str, optional): Absolute or relative path to the .pkl file.
                                  Defaults to chroma_documents.pkl in the
                                  same directory as this script.

    Returns:
        dict: {"status": "success", "code": 200, "message": "..."}
              or {"status": "error",   "code": 500, "message": "<error>"}
    """
    global _vectorizer, _tfidf_matrix, _archive, _pkl_path

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {
            "status": "error",
            "code": 500,
            "message": "scikit-learn is not installed. Run: pip install scikit-learn"
        }

    # Resolve path
    if pkl_path:
        _pkl_path = pkl_path

    # Fallback to chroma_documents.pkl if passed full state is not found
    if not os.path.exists(_pkl_path):
        fallback_path = os.path.join(os.path.dirname(_pkl_path), "chroma_documents.pkl")
        if os.path.exists(fallback_path):
            _pkl_path = fallback_path
        else:
            return {
                "status": "error",
                "code": 500,
                "message": f"PKL file not found at: {_pkl_path}"
            }

    try:
        # Load archive
        with open(_pkl_path, "rb") as f:
            _archive = pickle.load(f)

        # Validate archive keys
        required_keys = {"ids", "documents", "metadatas"}
        missing = required_keys - set(_archive.keys())
        if missing:
            return {
                "status": "error",
                "code": 500,
                "message": f"PKL archive is missing keys: {missing}"
            }

        # Build TF-IDF vectorizer and matrix
        _vectorizer = TfidfVectorizer(stop_words='english')
        _tfidf_matrix = _vectorizer.fit_transform(_archive["documents"])

        loaded_count = len(_archive["ids"])
        return {
            "status": "success",
            "code": 200,
            "message": f"Model loaded successfully. TF-IDF retriever ready with {loaded_count} items."
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
    Takes user input, retrieves relevant medical documents via TF-IDF search,
    and generates a structured medical response using Google Gemini.

    Args:
        input_data (dict): Must contain at minimum:
            - "query"          (str)  : The medical question.
            - "google_api_key" (str)  : Gemini API key (or set GEMINI_API_KEY/GOOGLE_API_KEY env var).
          Optionally:
            - "child_info"     (dict) : Child age and symptoms.
            - "parent_info"    (dict) : Parent observations.
            - "chat_history"   (list) : Previous conversation turns.
            - "n_results"      (int)  : Number of docs to retrieve (default 3).

    Returns:
        dict: Structured JSON response with status, message, and severity flags.
    """
    global _vectorizer, _tfidf_matrix, _archive

    # ---- Validate model is loaded ----
    if _vectorizer is None or _tfidf_matrix is None or _archive is None:
        return {
            "status": "error",
            "code": 500,
            "message": "Model not loaded. Call load() first.",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    # ---- Validate required inputs ----
    if not isinstance(input_data, dict):
        return {
            "status": "error",
            "code": 400,
            "message": "input_data must be a dictionary.",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    query = input_data.get("query", "").strip()
    if not query:
        return {
            "status": "error",
            "code": 400,
            "message": "Missing required field: 'query' must be a non-empty string.",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    # ---- Resolve API key ----
    api_key = input_data.get("google_api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return {
            "status": "error",
            "code": 400,
            "message": "Missing 'google_api_key' in input_data or GEMINI_API_KEY/GOOGLE_API_KEY environment variable.",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    # ---- Extract optional inputs ----
    child_info   = input_data.get("child_info",   {})
    parent_info  = input_data.get("parent_info",  {})
    chat_history = input_data.get("chat_history", [])
    n_results    = int(input_data.get("n_results", 3))

    def _sanitize(info):
        if not isinstance(info, dict): return info
        sanitized = info.copy()
        if "assessment" in sanitized and isinstance(sanitized["assessment"], dict):
            if "cluster_comparison" in sanitized["assessment"]:
                del sanitized["assessment"]["cluster_comparison"]
        if "lastResult" in sanitized and isinstance(sanitized["lastResult"], dict):
            if "cluster_comparison" in sanitized["lastResult"]:
                del sanitized["lastResult"]["cluster_comparison"]
        return sanitized

    child_info_clean = _sanitize(child_info)
    parent_info_clean = _sanitize(parent_info)

    try:
        # ----------------------------------------------------------------
        # Step 1: TF-IDF Search — Query TF-IDF matrix for top documents
        # ----------------------------------------------------------------
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = _vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:n_results]

        top_docs  = [_archive["documents"][idx] for idx in top_indices]
        top_metas = [_archive["metadatas"][idx] for idx in top_indices]

        # Build RAG context for prompt
        rag_context = []
        for doc, meta in zip(top_docs, top_metas):
            source = meta.get("source", "Unknown")
            rag_context.append({"source": source, "content": doc})

        # ----------------------------------------------------------------
        # Step 2: Generate response using Google Gemini
        # ----------------------------------------------------------------
        try:
            from google import genai
        except ImportError:
            return {
                "status": "error",
                "code": 500,
                "message": "google-genai is not installed. Run: pip install google-genai",
                "isSerious": False,
                "shouldCallEmergency": False
            }

        client = genai.Client(api_key=api_key)

        system_instruction = """
You are a professional medical first responder.
Analyze the provided information about a child and parent.
Use the provided medical research documents (RAG) to inform your response.

GUIDELINES:
1. In simple cases: give helpful suggestions.
2. Emergency: advise doctor immediately, set isSerious + shouldCallEmergency = true.
3. Suspicious: suggest doctor, set isSerious = true.
4. Use chat history.
5. Output ONLY JSON.
"""

        user_prompt = f"""
INPUT DATA:
- Current Question: {json.dumps(query)}
- Child Info: {json.dumps(child_info_clean)}
- Parent Info: {json.dumps(parent_info_clean)}
- Reference Docs (RAG): {json.dumps(rag_context)}
- Chat History: {json.dumps(chat_history)}

OUTPUT JSON:
{{
    "status": "success/error",
    "code": 200,
    "message": "...",
    "isSerious": true/false,
    "shouldCallEmergency": true/false
}}
"""

        gen_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "system_instruction": system_instruction
        }

        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=user_prompt,
            config=gen_config
        )

        try:
            result = json.loads(response.text, strict=False)
        except json.JSONDecodeError:
            # Fallback if strict=False still fails due to weird formatting
            import re
            cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response.text)
            result = json.loads(cleaned_text)

        # ---- Enrich result with RAG metadata ----
        result["rag_documents"] = top_docs
        result["sources"] = [m.get("source", "") for m in top_metas]
        if "code" not in result:
            result["code"] = 200

        return result

    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "code": 500,
            "message": f"Failed to parse Gemini response as JSON: {str(e)}",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    except KeyError as e:
        return {
            "status": "error",
            "code": 400,
            "message": f"Missing or invalid input field: {str(e)}",
            "isSerious": False,
            "shouldCallEmergency": False
        }

    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": str(e),
            "isSerious": False,
            "shouldCallEmergency": False
        }


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly to verify)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== chatbot_model.py self-test ===\n")

    print("[1] Loading model...")
    load_result = load()
    print(json.dumps(load_result, indent=2))

    if load_result["code"] == 200:
        print("\n[2] Running a sample prediction...")
        sample_input = {
            "query": "What is whooping cough?",
            "child_info": {"age": "5", "symptoms": "Mild fever and cough"},
            "parent_info": {"observation": "Seems lethargic but is drinking water"},
            "chat_history": [],
            "n_results": 3,
            "google_api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
        }
        result = predict(sample_input)
        print(json.dumps(result, indent=2))
    else:
        print("Skipping predict() test — model failed to load.")
