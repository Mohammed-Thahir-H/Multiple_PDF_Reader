import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            text+= pages.extract_text()
        return text
    
def get_text_chunks(text):
    text_splippter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap= 200,
        length_function = len
    )
    chunks = text_splippter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    # embeddings= OpenAIEmbeddings()
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorestore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorestore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_restriver(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Reader")

    if "conversation" not in st.session_state:
        st.session_state

    st.header("Chat with your own PDF :books:")
    st.text_input("Ask me your question:")
    

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs and click on Submit", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing"):
                raw_text= get_pdf_text(pdf_docs)
                # st.write(raw_text)

                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                vectorstore = get_vectorestore(text_chunks)
                # st.write(vectorstore)

                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation

if __name__ == '__main__' :
    main()