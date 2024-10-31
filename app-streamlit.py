import os
from helper_functions.youtube_tools import YouTubeVideoLoader
from langchain_community.tools import DuckDuckGoSearchResults
from helper_functions.text_processing import extract_links_from_string
from langchain_community.document_loaders import WebBaseLoader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type, List
import requests
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
import streamlit as st

load_dotenv()

# Initialize an empty string for the combined transcript
transcript = ""
product = st.text_input("Enter the name of product")
query=""

if product and not query:  # Only execute the following code if a product name is entered.
    user_query = f"{product} reviews"
    number_of_transcripts = 5
    
    # Load YouTube video transcripts based on the product query
    youtube_loader = YouTubeVideoLoader(user_query, max_results=number_of_transcripts)

    # Retrieve and concatenate transcripts from YouTube videos
    for idx, video_transcript in enumerate(youtube_loader.transcripts):
        print(f"Transcript for Video ID {youtube_loader.video_ids[idx]}:")
        for entry in video_transcript:
            transcript += entry['text'] + "\n"  # Add newline for readability

    tool = DuckDuckGoSearchResults(output_format="list")
    results = tool.run(user_query)
    parsed_links = extract_links_from_string(results)

    # Scrape each link and concatenate their content
    loader = WebBaseLoader(parsed_links)
    documents = loader.load()
    transcript += "\n".join(doc.page_content for doc in documents) + "\n"  # Add newline for readability

    # Export string as .txt file
    os.makedirs("temp_data", exist_ok=True)

    # Define the full path for the output file
    file_path = os.path.join("temp_data", "scrapped_data.txt")  # Ensure .txt extension is included

    # Write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f"Data successfully exported to {file_path}")

    ############################################################################################################

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Function to load text data from a file
    def load_text_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Function to create FAISS index from text data
    def create_faiss_index(text):
        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        # Create embeddings for each chunk
        embeddings = embedding_model.encode(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        return index, chunks

    # Load the scrapped data and create FAISS index
    file_path = os.path.join("temp_data", "scrapped_data.txt")
    text_data = load_text_data(file_path)
    faiss_index, text_chunks = create_faiss_index(text_data)

    query = st.text_input(f"Ask any question about {product}:")

    ################################################################################################################
elif product and query:
    # Tool to perform RAG

    class GetFAISSInput(BaseModel):
        query: str = Field(description="The search query to retrieve relevant information from database.")
        max_results: int = Field(default=4, description="The maximum number of chunks to return.")

    class GetFAISSTool(BaseTool):
        name: str = "faiss_tool"
        description: str = "Useful to get usage and review based data from youtube and websites."
        args_schema: type = GetFAISSInput  # Specify the input schema

        def _run(self, query: str, max_results: int=4) -> str:
            question_embedding = embedding_model.encode([query])
        
            # Search for relevant chunks in the FAISS index
            distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), k=max_results)
            relevant_chunks = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]

            llm= ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
            input_text=f"""You are an assistant for question-answering tasks. 
                            You have been given name of a product {product} and text chunks based on 
                            user query. Clean the text, 
                            understand the context using {query} and rewrite the data in chunk as per your 
                            understanding and give an output\n{relevant_chunks}
                            Please ensure to return only relevant text and no other text.
                            If you don't know the answer, say that you don't know."""

            response = llm.invoke(input_text)
            return response.content


    # Tool for Tabily 
    class TabilySearchInput(BaseModel):
        query: str = Field(description="should be a search query")

    class TabilySearchTool(BaseTool):
        name: str= "simple_search"
        description: str = "Useful to get Product details and reviews from internet"
        args_schema: Type[BaseModel] = TabilySearchInput

        def _run(self, query: str) -> str:
            """Use the tool."""
            from tavily import TavilyClient

            api_key = os.getenv("TAVILY_API_KEY")
            client = TavilyClient(api_key=api_key)
            results = client.search(query=query)
            return f"Search results for: {query}\n\n\n{results}\n"


    tools=[TabilySearchTool(), GetFAISSTool()]

    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.4)
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    response = agent_executor.invoke({"input": f" Use your tools to answer the following query \n{query}"})

    st.write("Response:", response)