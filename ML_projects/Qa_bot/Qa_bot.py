from PyPDF2 import PdfReader
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text




def get_text_chunks(text):
    text_spiltter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_spiltter.split_text(text)
    return chunks




def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts = text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
    


# def get_conversation_chain():
#     prompt_template =  """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(repo_id="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template= prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
#     return chain





# def handle_userinput(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index",embeddings)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversation_chain()
#     response = chain({"input_documents":docs, "question": user_question}, return_only_outputs = True)
#     print(response)
#     st.write("Reply: ", response["output_text"])
   
    
    
    

def main():
    
    
    st.set_page_config(page_title="PdfPal", page_icon=":page_with_curl:")
    
    user_question = st.text_input("Enter your querry here!")
    
    # if user_question:
    #     handle_userinput(user_question)
    
    st.header("Chat with Pdfs Using GEMINI :page_with_curl:")
    
    with st.sidebar:
        st.subheader("Your documnets")
        pdf_docs = st.file_uploader("Upload your files here and click on 'Upload'",accept_multiple_files=True)
        if st.button("Upload") :
           with st.spinner("Processing"):
                #get pdf text 
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text) {test}
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)  {test}
                #create the vector store
                get_vector_store(text_chunks)
                #create conversation chain 
                st.success("done")
            
   
   
                
if __name__ == '__main__':
    main()