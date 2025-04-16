import streamlit as st
from rag_function import rag

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .stTextInput {
        padding: 10px;
    }
    .stMarkdown {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# set initial message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}
    ]

# Display chat header
st.title("💬 AI Assistant")
st.markdown("Ask me anything! I'll try my best to help you.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag(prompt)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})