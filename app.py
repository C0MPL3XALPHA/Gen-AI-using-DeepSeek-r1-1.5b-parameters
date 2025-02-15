import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.title("Your AI companion!")
with st.sidebar:
 st.header("⚙️ Configuration")
 selected_model=st.selectbox("Choose Model",["deepseek-r1:1.5b","deepseek-r1:7b"])
 st.divider()
 st.subheader("Model Capabilities")
 st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Documentation
    - 💡 Solution Design
    """)
 st.divider()
 st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/) | [DeepSeek](https://www.deepseek.com/)")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")



llm=ChatOllama(model=selected_model,temperature=0.15,base_url="http://localhost:11434/")

#system prompt configuration
system_prompt=SystemMessagePromptTemplate.from_template("You are an Computer Vision Expert.You have the knowledge of all the current developments on Image Super resolution. Provide logics and also great explanations. "
                                                        "Always respond in English. And be to the point")

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()