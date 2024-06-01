Multilingual Retrieval-Augmented Generation (RAG) System
This repository contains a Multilingual Retrieval-Augmented Generation (RAG) system that transcribes audio from video or audio files, generates embeddings, and creates a Pinecone vector index for efficient retrieval. The system utilizes OpenAI's Whisper for transcription, Sentence Transformer for embeddings, and integrates with GPT-3.5 via Langchain for summarization and translation tasks. Evaluation of the generated summaries and translations is performed using BLEU and ROUGE metrics.

Getting Started
Requirements
Install the required Python libraries:

sh
Copy code
pip install -r requirements.txt
requirements.txt:

Copy code
moviepy
openai-whisper
sentence-transformers
pinecone-client
langchain
nltk
rouge-score
Setup
Clone the repository:

sh
Copy code
git clone https://github.com/your-username/multilingual-rag-system.git
cd multilingual-rag-system
Set up environment variables:
Create a .env file and add your API keys:

makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
Download the translation dataset:
Ensure you have a translations dataset and preprocess it to create embeddings.
