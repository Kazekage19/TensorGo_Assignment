# Multilingual Retrieval-Augmented Generation (RAG) System
# Submission by Akshat Srivastava - 2020UEA6620 NSUT
This repository contains a Multilingual Retrieval-Augmented Generation (RAG) system that transcribes audio from video or audio files, generates embeddings, and creates a Pinecone vector index for efficient retrieval. The system utilizes OpenAI's Whisper for transcription, Sentence Transformer for embeddings, and integrates with GPT-3.5 via Langchain for summarization and translation tasks. Evaluation of the generated summaries and translations is performed using BLEU and ROUGE metrics.

## Getting Started

### Requirements

Install the required Python libraries:

```sh
pip install -r requirements.txt
```
Setup
Clone the repository:

```sh
git clone https://github.com/Kazekage19/TensorGo_Assignment.git
cd TensorGo_Assignment
```
Set up environment variables:

Create a .env file and add your API keys:

```sh
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
Download the translation dataset:

Ensure you have a translations dataset and preprocess it to create embeddings.

#Methodology
1)Extract Audio:

Use movie.py to extract audio from a video file. If you have an audio file, use it directly.


2)Transcribe Audio:

I used OpenAI Whisper to transcribe the audio.


3) Used Sentence Transformer to create embeddings for the transcriptions and the translation dataset.

4) Then created Pinecone Vector Index .Used Pinecone to create a vector index of these embeddings.

5) Integrated with Langchain for RAG, Langchain with GPT-3.5 and the Pinecone index retriever to act as a RAG system and query for summaries and translations of the transcribed audio.

6) Finally, metrics evaluation was done by evaluating the ROUGE and BLEU scores for the generated summaries and translations.
