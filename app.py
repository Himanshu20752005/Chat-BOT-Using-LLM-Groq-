# Import necessary libraries
from dotenv import load_dotenv  # For loading environment variables from .env file
import os  # For accessing system environment variables
import streamlit as st  # For building the web UI
from langchain.chains import ConversationChain  # For handling AI conversation flow
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  # For maintaining conversational memory
from langchain_groq import ChatGroq  # For connecting with Groq's LLM

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key for Groq from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is available, if not, show an error in Streamlit and stop execution
if not groq_api_key:
    st.error("Error: GROQ_API_KEY is missing. Check your .env file.")
    st.stop()  # Stop execution to prevent further errors

# Define the main function for the Streamlit app
def main():
    # Set the title of the web app
    st.title("Coding Mentor")

    # Sidebar section for selecting model and settings
    st.sidebar.title("Group No 3")  # Sidebar title
    st.sidebar.subheader("Select an LLM")  # Subtitle for model selection
    
    # Dropdown to choose between two available LLM models
    model = st.sidebar.selectbox(
        "Choose a model",
        ["mixtral-8x7b-32768", "llama2-70b-4096"]
    )

    # Slider to select the memory length (how many past messages the chatbot remembers)
    conversational_memory_length = st.sidebar.slider("Conversational memory length:", 1, 10, value=5)

    # Initialize conversational memory with the selected memory length
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Create an empty list to store chat messages

    else:
        # If chat history exists, restore previous conversations in memory
        for message in st.session_state.chat_history:
            if "human" in message and "AI" in message:  # Ensure both keys exist
                memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    # Initialize Groq Chat model with the selected LLM
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,  # API key to authenticate requests
        model_name=model  # Selected model
    )

    # Set up conversation handling using Langchain's ConversationChain
    conversation = ConversationChain(
        llm=groq_chat,  # Connects to the Groq chatbot
        memory=memory  # Stores the short-term conversation memory
    )

    # Input box for users to type their questions
    user_question = st.text_area("Ask a question:")

    if user_question:  # Check if user entered a question
        # Get AI's response by passing the user question to the chatbot
        response = conversation.predict(input=user_question)

        # Store the conversation in chat history (for memory)
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)

        # Display chatbot's response
        st.write("Chatbot:", response)

# Run the main function when executing the script
if __name__ == "__main__":
    main()
