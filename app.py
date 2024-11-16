import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import ImageCaptionLoader
from dotenv import load_dotenv
import os
import base64
from PIL import Image
from prompt import system_prompt

# Load environment variables
load_dotenv("var.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model and retriever
model = ChatOpenAI(model="gpt-4o", temperature=0)


# Initialize Chat Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Define the Streamlit UI
st.title("Health & Food Analyzer")
st.write("Upload an image of your food, fruit, or drink, and describe your health condition to get personalized advice!")

# Initialize session state for image persistence
if "uploaded_file_data" not in st.session_state:
    st.session_state["uploaded_file_data"] = None
if "uploaded_file_image" not in st.session_state:
    st.session_state["uploaded_file_image"] = None

# User Inputs
health_condition = st.text_input(
    "Describe your health condition (e.g., I have a fever):")
uploaded_file = st.file_uploader(
    "Upload an image of your food, fruit, or drink:", type=["jpg", "jpeg", "png"])

# Process on Upload
if uploaded_file:
    st.session_state["uploaded_file_data"] = uploaded_file.read()
    st.session_state["uploaded_file_image"] = Image.open(uploaded_file)

# Display the uploaded image if available
if st.session_state["uploaded_file_image"]:
    st.image(st.session_state["uploaded_file_image"],
             caption="Uploaded Image", use_container_width=True)

if st.session_state["uploaded_file_data"] and health_condition:
    try:
        # Read and validate the uploaded file
        buffered = st.session_state["uploaded_file_data"]
        if len(buffered) < 10:
            st.error(
                "The uploaded file seems to be empty or corrupted. Please upload a valid image.")
        else:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
                temp_file.write(buffered)
                temp_file_path = temp_file.name

            # Load image with ImageCaptionLoader
            loader = ImageCaptionLoader(images=[temp_file_path])
            list_docs = loader.load()

            # Process and analyze
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(list_docs)
            vectorstore = Chroma.from_documents(
                documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever(k=2)

            # Create and run RAG chain
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(
                retriever, question_answer_chain)
            input_query = f"My health condition is: {health_condition}. Analyze if the uploaded item is suitable for me."
            response = rag_chain.invoke({"input": input_query})

            st.subheader("Analysis Result:")
            st.write(response["answer"])

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please provide both your health condition and an image for analysis.")
