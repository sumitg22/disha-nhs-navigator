import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

st.set_page_config(
    page_title="Disha — NHS Navigator",
    page_icon="🌟",
    layout="centered"
)

st.title("🌟 Disha")
st.subheader("Your NHS Navigation Assistant")
st.caption("Helping South Asians navigate healthcare in the UK")
st.markdown("---")

# Load and build RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    # Load NHS pages
    urls = [
        "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/",
        "https://www.nhs.uk/nhs-services/urgent-and-emergency-care-services/when-to-use-111/",
        "https://www.nhs.uk/mental-health/nhs-voluntary-charity-services/nhs-services/",
        "https://www.nhs.uk/nhs-services/students/",
        "https://www.nhs.uk/using-the-nhs/healthcare-abroad/moving-to-england/how-to-access-nhs-services-in-england/"
    ]

    loader = WebBaseLoader(urls)
    documents = loader.load()

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Connect Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Create prompt
    prompt_template = PromptTemplate.from_template("""You are Disha, a warm and 
helpful AI assistant helping South Asians navigate NHS and healthcare services 
in the UK. Use the following context from official NHS documents to answer 
the question. If you don't know the answer from the context, say "I don't have 
that information in my NHS documents — please visit nhs.uk for more details."

Context: {context}

Question: {question}

Answer in a warm, clear, and helpful tone:""")

    # Create RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return qa_chain

# Load pipeline
with st.spinner("Disha is getting ready..."):
    qa_chain = load_rag_pipeline()

st.success("Disha is ready to help you!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Namaste! 🙏 I am Disha, your NHS navigation assistant. I can help you with GP registration, NHS 111, mental health services, student healthcare, and more. What would you like to know?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about NHS services..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get Disha's response
    with st.chat_message("assistant"):
        with st.spinner("Disha is thinking..."):
            response = qa_chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

# Sidebar
with st.sidebar:
    st.header("About Disha")
    st.write("Disha helps South Asians navigate NHS services in the UK using official NHS documents.")
    st.markdown("---")
    st.write("**You can ask about:**")
    st.write("• GP registration")
    st.write("• NHS 111 service")
    st.write("• Mental health support")
    st.write("• Student healthcare")
    st.write("• Emergency services")
    st.markdown("---")
    st.caption("Built by Sumit | Project Disha")
