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
