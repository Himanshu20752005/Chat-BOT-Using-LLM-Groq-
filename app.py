from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Error: GROQ_API_KEY is missing. Check your .env file.")
    st.stop()

# Streamlit app
def main():
    st.title("Coding Mentor")

    # Sidebar for model selection and settings
    st.sidebar.title("Group No 3")
    st.sidebar.subheader("Select an LLM")
    
    model = st.sidebar.selectbox(
        "Choose a model",
        ["mixtral-8x7b-32768", "llama2-70b-4096"]
    )
    conversational_memory_length = st.sidebar.slider("Conversational memory length:", 1, 10, value=5)

    # Conversation Memory
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    else:
        for message in st.session_state.chat_history:
            if "human" in message and "AI" in message:
                memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    # Initialize Groq Chat
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Conversation Chain
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # User Input
    user_question = st.text_area("Ask a question:")

    if user_question:
        response = conversation.predict(input=user_question)
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
