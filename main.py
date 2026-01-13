import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Streamlit framework
st.title("Q & A assistant")
input_text = st.text_input("Search the topic you want")

# Initialize chat message history for Streamlit
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Initialize the OpenRouter LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.8,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)

# Create prompt with message history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides detailed explanations and then simplifies them."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Provide a detailed explanation about: {topic}"),
    ("human", "Now explain that in simple terms.")
])

# Create chain
chain = prompt | llm | StrOutputParser()

# Wrap chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="topic",
    history_messages_key="history",
)

if input_text:
    with st.spinner("Generating response... ⏳"):
        response = chain_with_history.invoke(
            {"topic": input_text},
            config={"configurable": {"session_id": "any"}}
        )
        
    st.write(response)
    
    # Display chat history
    if st.checkbox("Show conversation history"):
        st.write("### Chat History")
        for msg in msgs.messages:
            st.write(f"**{msg.type}**: {msg.content}")