from helper_functions.youtube_tools import YouTubeVideoLoader
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def process_youtube_query(product_name, number_of_transcripts=5):
    def load_transcripts(query, number_of_transcripts):
        youtube_loader = YouTubeVideoLoader(query, max_results=number_of_transcripts)
        
        transcripts = ""
        for idx, transcript in enumerate(youtube_loader.transcripts):
            print(f"Transcript for Video ID {youtube_loader.video_ids[idx]}:")
            for entry in transcript:
                transcripts += entry['text'] + "\n"  # Add newline for readability

        print(f"Total transcripts retrieved: {len(youtube_loader.transcripts)}") if transcripts else print("No transcripts retrieved.")
        return transcripts

    def create_faiss_index(transcripts):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        
        chunks = text_splitter.split_text(transcripts)
        vectors = model.encode(chunks)
        dimension = vectors.shape[1]
        
        index = faiss.IndexFlatL2(dimension)

        if len(vectors) > 0:
            index.add(np.array(vectors).astype('float32'))  # Ensure vectors are in float32 format
        
        return index, chunks

    def save_index_and_transcripts(index, chunks, transcripts):
        os.makedirs('temp_data', exist_ok=True)

        # Save transcripts to a text file
        with open(os.path.join('temp_data', 'transcripts.txt'), 'w', encoding='utf-8') as f:
            f.write(transcripts)

        # Save FAISS index to a file
        faiss.write_index(index, os.path.join('temp_data', 'faiss_index.index'))

    # Load transcripts and create FAISS index
    transcripts = load_transcripts(product_name, number_of_transcripts)
    
    if transcripts:
        index, chunks = create_faiss_index(transcripts)
        save_index_and_transcripts(index, chunks, transcripts)
        print("Transcripts and FAISS index have been saved successfully in 'temp_data'.")

# # Example usage:
# if __name__ == "__main__":
#     process_youtube_query("Acer Aspire 7 review")