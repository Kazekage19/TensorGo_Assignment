import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import whisper
import openai
from transcribe import extract_audio
openai.api_key = "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"
model = whisper.load_model("base")
import os
print("Current Working Directory:", os.getcwd())
if os.path.isfile("sample.mp3"):
    print("sample.mp3 exists in the current directory.")
else:
    print("sample.mp3 does NOT exist in the current directory.")
file_path = "Replace_Your_FilePath"
audio_path = extract_audio(file_path)
print("Absolute Path:", audio_path)
model = whisper.load_model("large")
result = model.transcribe(audio_path, task="transcribe")
transcription = result["text"]
print("Transcription:", transcription)

embed_model = SentenceTransformer('distiluse-base-multilingual-cased')
transcriptions = [transcription]
trans_embed = embed_model.encode(transcriptions)
df = pd.read_csv(r'corpus.csv',nrows=1000)
texts = df['text'].tolist()
lang_embed= embed_model.encode(texts)
data_for_db =[
    {"id": f" Translations- {i}", "values" : lang_embed[i], "metadata" : {"text" : texts[i] ,"language" : df['language'][i]}}
    for i in range(len(texts))
]
Documents =[
    {'page_content' : texts[i], "metadata" : {"language" : df['language'][i]}}
    for i in range(len(texts))
]
api_key = '7b1979e5-6bd7-48f8-918a-6077994d81e5'
os.environ["PINECONE_API_KEY"] = api_key
pc = Pinecone(api_key=api_key)
index_name = "tensorgo-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
index.upsert(data_for_db)

def queries(vectors, query):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain_pinecone import PineconeVectorStore
    from langchain_core.embeddings.embeddings import Embeddings
    # embeddings = Pinecone(index_name=index_name, embedding_function=embed_model.encode) 
    # documents = [{"page_content": transcription}] 
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key= "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")
    # docsearch = PineconeVectorStore.from_documents(documents=Documents, embedding=lang_embed, index_name=index)
    class embedding_function():
      def __init__(self):
        pass
      def embed_query(self,texts):
        return embed_model.encode(texts)
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name,embedding = embedding_function)
    retriever= docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'lambda_mult': 0.25})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(query)
    return answer

question = f'Summarize the text: \n {transcription}' 
summary = queries(index, question)
print(summary)

ques = f'Translate the text into French: \n {transcription}'
translation = queries(index, ques)
print(translation)


#Here I will do the evaluation
ref_doc ="Insert_file_for_correct_translation"
import evaluation
ref_doc = evaluation.load_reference_translations(ref_doc)
bleu = evaluation.evaluate_bleu(ref_doc , summary)
rouge = evaluation.evaluate_rouge(ref_doc , summary)

print("bleu score" , bleu)
print("ROUGE Score" , rouge )

