import streamlit as st
from RAG import process_pdf_to_pinecone, rag_agent
from langchain_core.messages import HumanMessage, AIMessage
import os

st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize session state
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Custom CSS for modern chat interface
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        margin-bottom: 10px;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 15px;
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 15px;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 18px;
        word-wrap: break-word;
    }
    
    .user-bubble {
        background-color: #007bff;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .bot-bubble {
        background-color: #2d3748;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 8px;
        font-size: 14px;
    }
    
    .user-avatar {
        background-color: #007bff;
        color: white;
    }
    
    .bot-avatar {
        background-color: #4a5568;
        color: white;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #4a5568 !important;
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
        padding: 12px 20px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns: sidebar and main chat area
col1, col2 = st.columns([1, 3])

# Sidebar for PDF upload
with col1:
    st.markdown("### 📄 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to chat with its contents"
    )
    
    if uploaded_file is not None:
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            if not st.session_state.processing:
                # Save uploaded file
                file_path = os.path.join("data", uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process PDF
                    st.session_state.processing = True
                    with st.spinner("🔄 Processing PDF and creating embeddings..."):
                        process_pdf_to_pinecone(file_path)
                    
                    # Update session state
                    st.session_state.pdf_uploaded = True
                    st.session_state.chat_history = []  # Clear previous chat
                    st.session_state.last_uploaded_filename = uploaded_file.name
                    st.session_state.processing = False
                    
                    st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.processing = False
            else:
                st.info("🔄 Processing in progress...")
    
    # Show current status
    if st.session_state.pdf_uploaded and st.session_state.last_uploaded_filename:
        st.success(f"📖 Current PDF: {st.session_state.last_uploaded_filename}")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("📤 Please upload a PDF to start chatting")

# Main chat area
with col2:
    st.markdown("### 💬 Chat with your PDF")
    
    if st.session_state.pdf_uploaded:
        # Chat display container
        chat_placeholder = st.empty()
        
        # Display chat history
        with chat_placeholder.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if not st.session_state.chat_history:
                st.markdown("""
                    <div style="text-align: center; color: #6c757d; padding: 50px;">
                        <h4>👋 Hello! I'm ready to help you with your PDF.</h4>
                        <p>Ask me anything about the document you've uploaded!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                for i, message in enumerate(st.session_state.chat_history):
                    if isinstance(message, HumanMessage):
                        st.markdown(f"""
                            <div class="user-message">
                                <div class="message-bubble user-bubble">{message.content}</div>
                                <div class="message-avatar user-avatar">👤</div>
                            </div>
                        """, unsafe_allow_html=True)
                    elif isinstance(message, AIMessage):
                        st.markdown(f"""
                            <div class="bot-message">
                                <div class="message-avatar bot-avatar">🤖</div>
                                <div class="message-bubble bot-bubble">{message.content}</div>
                            </div>
                        """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Message",
                    placeholder="Type your question here...",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col_button:
                submit_button = st.form_submit_button("Send", use_container_width=True)
            
            if submit_button and user_input.strip():
                # Add user message to chat history
                user_message = HumanMessage(content=user_input.strip())
                st.session_state.chat_history.append(user_message)
                
                # Get response from RAG agent
                try:
                    with st.spinner("🤔 Thinking..."):
                        # Prepare messages for the agent
                        messages_for_agent = st.session_state.chat_history.copy()
                        
                        # Invoke the RAG agent
                        result = rag_agent.invoke({"messages": messages_for_agent})
                    
                    # Check if tool was called and show appropriate spinner
                    if result and 'messages' in result and result['messages']:
                        # Check if any tool messages are present
                        tool_called = any(
                            hasattr(msg, '__class__') and 'Tool' in msg.__class__.__name__ 
                            for msg in result['messages']
                        )
                        
                        if tool_called:
                            with st.spinner("🔧 Calling tool to search PDF..."):
                                # Small delay to show the tool calling message
                                import time
                                time.sleep(0.5)
                        
                        # Get the last message which should be the AI response
                        ai_response = result['messages'][-1]
                        st.session_state.chat_history.append(ai_response)
                    else:
                        # Fallback if result structure is unexpected
                        error_message = AIMessage(content="Sorry, I couldn't process your request. Please try again.")
                        st.session_state.chat_history.append(error_message)
                
                except Exception as e:
                    error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
                    st.session_state.chat_history.append(error_message)
                
                # Rerun to update the chat display
                st.rerun()
    
    else:
        # Show message when no PDF is uploaded
        st.markdown("""
            <div style="text-align: center; padding: 100px; color: #6c757d;">
                <h3>📄 No PDF Uploaded</h3>
                <p>Please upload a PDF document in the sidebar to start chatting.</p>
                <p>Once uploaded, you can ask questions about the document content!</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.8em;">
        💡 <strong>Tip:</strong> Ask specific questions about your PDF content for the best results!
    </div>
""", unsafe_allow_html=True)