


# PDF_AI 

This repository contains a PDF chatbot application built using Streamlit, LangChain, HuggingFace, and various other libraries. The application allows users to upload multiple PDFs and interact with their content through a conversational interface.

## Features

- **PDF Text Extraction**: Extracts text from uploaded PDF documents.
- **Text Chunking**: Splits extracted text into manageable chunks for processing.
- **Embeddings and Vector Store**: Creates embeddings using HuggingFace models and stores them in a vector store for efficient retrieval.
- **Conversational AI**: Uses a conversational AI model to answer user queries based on the content of the uploaded PDFs.
- **Interactive UI**: Streamlit-based user interface for easy interaction.

## Libraries Used

- `streamlit`: Web application framework for creating interactive UIs.
- `PyPDF2`: Library for reading and extracting text from PDFs.
- `langchain`: Provides utilities for text splitting, embeddings, and conversational AI.
- `huggingface-hub`: Access to HuggingFace models and token authentication.
- `faiss-cpu`: Efficient similarity search and clustering of dense vectors.
- `altair`: Declarative statistical visualization library.
- `openai`, `tiktoken`, `InstructorEmbedding`, `sentence-transformers`: Additional libraries for embeddings and language models.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/pdf_ai_chatbot.git
    cd pdf_ai_chatbot
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Setup

Make sure to authenticate with HuggingFace to access models:

```python
from huggingface_hub import login
login(token='your_hugging_face_auth_token')
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

## Code Overview

### PDF Text Extraction

Extracts text from uploaded PDF documents.

```python
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```

### Text Chunking

Splits the extracted text into manageable chunks.

```python
from langchain.text_splitter import CharacterTextSplitter

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
```

### Embeddings and Vector Store

Creates embeddings using HuggingFace models and stores them in a vector store.

```python
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```

### Conversational AI

Sets up a conversational AI model to interact with the extracted PDF content.

```python
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
```

### Main Application

Runs the Streamlit application and handles user input.

```python
import streamlit as st

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

if __name__ == "__main__":
    main()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The HuggingFace team for providing access to various language models.
- The Streamlit team for creating an excellent framework for building interactive web apps.
- The LangChain library for providing tools to build advanced NLP applications.

---

Feel free to modify the README as per your specific requirements or project details.
