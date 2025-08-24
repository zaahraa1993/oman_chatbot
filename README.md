Oman Investment Chatbot

This project is a simple Retrieval-Augmented Generation (RAG) chatbot that answers questions about Oman Vision 2040 and related investment opportunities.

Backend: Python + FastAPI

Retrieval: Vector similarity search over a custom dataset

LLM: Uses a base language model (no fine-tuning yet)

Deployment: Tested on a VPS with Gunicorn

⚠️ Since the model is not fine-tuned, the chatbot only retrieves answers from the dataset. If the question is not covered, it will respond with:

I don't know. Please consult a domain expert.
Since the retrieval sometimes returns similar but not exact matches, the model may occasionally provide irrelevant or approximate answers.

The current dataset is a demo version, covering selected Q&A about Oman Vision 2040. It’s sufficient to demonstrate the RAG pipeline, but I’m preparing a more comprehensive dataset for fine-tuning and improved accuracy.

This project is deployed on a VPS using **Gunicorn** and **FastAPI**.

### API Endpoint
http://145.79.12.66:8081/chat

### Method  
POST

### Request Body (JSON)
```json
{
  "question": "Tell me about the Duqm Port investment opportunity."
}


Example Response

{
    "answer": "Port of Duqm is receiving a $550 million FDI, focusing on marine upgrades and green steel production, aligned with Oman’s Vision 2040.",
    "sources": [
        {
            "snippet": "Q: What can you tell me about the Duqm Port investment opportunity?\nA: Port of Duqm is receiving a $550 million FDI, focusing on marine upgrades and green steel production, aligned",
            "metadata": {
                "topic": "Tell me about the Duqm Port investment opportunity.",
                "question": "What can you tell me about the Duqm Port investment opportunity?"
            }
        },
        {
            "snippet": "Q: Tell me about the Duqm Port investment opportunity.\nA: Port of Duqm is receiving a $550 million FDI, focusing on marine upgrades and green steel production, aligned with Oman’s ",
            "metadata": {
                "topic": "Tell me about the Duqm Port investment opportunity.",
                "question": "Tell me about the Duqm Port investment opportunity."
            }
        },
        {
            "snippet": "Q: Tell me about What’s the expected investment in the heritage and tourism sector by 2025.\nA: For 2021–2025, the heritage and tourism sector investment is expected to reach OMR 3 ",
            "metadata": {
                "topic": "What’s the expected investment in the heritage and tourism sector by 2025?",
                "question": "Tell me about What’s the expected investment in the heritage and tourism sector by 2025."
            }
        },
        {
            "snippet": "Q: What can you tell me about What’s the expected investment in the heritage and tourism sector by 2025?\nA: For 2021–2025, the heritage and tourism sector investment is expected to",
            "metadata": {
                "topic": "What’s the expected investment in the heritage and tourism sector by 2025?",
                "question": "What can you tell me about What’s the expected investment in the heritage and tourism sector by 2025?"
            }
        }
    ]
}
    }
  ]
}
