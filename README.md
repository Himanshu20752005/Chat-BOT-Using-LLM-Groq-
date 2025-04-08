# Coding Mentor Chatbot

A Streamlit-based chatbot powered by Groq LLM models, designed to assist with coding-related queries.

## Features
- Supports multiple LLM models (`mixtral-8x7b-32768`, `llama2-70b-4096`).
- Maintains conversational context using `ConversationBufferWindowMemory`.
- User-adjustable conversational memory length.
- Simple UI built with Streamlit.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install streamlit langchain langchain_groq groq python-dotenv
```

## Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/python-chatbot.git
   cd python-chatbot
   ```

2. Create a `.env` file and add your Groq API key:

   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Run the chatbot:

   ```bash
   streamlit run chatbot.py
   ```

## Usage
- Enter your question in the text area.
- Select an LLM model from the sidebar.
- Adjust conversational memory length as needed.
- View the chatbot's responses in real-time.

## Project Structure
```
├── chatbot.py      # Main chatbot script
├── .env            # API key storage (not included in repo)
├── requirements.txt # Dependencies
├── README.md       # Documentation
```

## License
This project is licensed under the MIT License.

## Author
[Himanshu Pandey](https://github.com/Himanshu20752005)
