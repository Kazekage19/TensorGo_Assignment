# Multilingual Retrieval-Augmented Generation (RAG) System

This repository contains a Multilingual Retrieval-Augmented Generation (RAG) system that transcribes audio from video or audio files, generates embeddings, and creates a Pinecone vector index for efficient retrieval. The system utilizes OpenAI's Whisper for transcription, Sentence Transformer for embeddings, and integrates with GPT-3.5 via Langchain for summarization and translation tasks. Evaluation of the generated summaries and translations is performed using BLEU and ROUGE metrics.

## Getting Started

### Requirements

Install the required Python libraries:

```sh
pip install -r requirements.txt
