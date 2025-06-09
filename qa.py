import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import os
from dotenv import load_dotenv

# --- LangChain imports for Q&A ---
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="OCR with Llama Vision + Q&A",
    page_icon="📄🤖",
    layout="wide"
)

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def extract_text_with_llama(image, api_key):
    """Extract text from image using Llama-3.2-90B-Vision-Instruct-Turbo via Together AI"""
    
    # Convert image to base64
    base64_image = encode_image_to_base64(image)
    
    # Together AI API endpoint
    url = "https://api.together.xyz/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload
    payload = {
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract ALL the text from this image. Do NOT include any commentary or explanations—just return the extracted text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.1  # Low temperature for more precise extraction
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        extracted_text = result['choices'][0]['message']['content']
        return extracted_text, None
        
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {str(e)}"
    except KeyError as e:
        return None, f"Unexpected response format: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    st.title("📄 OCR with Llama Vision + Q&A")
    st.markdown("Extract text from images and then ask questions about that text using LangChain and Llama-3.2-90B-Vision-Instruct-Turbo")
    
    # Initialize API key variable
    api_key = ""
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        
        # Option to load API key from environment
        if "TOGETHER_API_KEY" in os.environ:
            api_key = os.getenv("TOGETHER_API_KEY")
            st.markdown("**API Key loaded from environment variable.**")
        else:
            # Manual entry
            manual_api_key = st.text_input(
                "Enter your Together AI API Key",
                type="password",
                placeholder="sk-...",
                help="Your API key is used to authenticate with the Llama-3.2-90B-Vision-Instruct-Turbo endpoint."
            )
            if manual_api_key:
                # Option 1: Use session state for persistence
                if st.checkbox("Remember API key for this session"):
                    st.session_state.api_key = manual_api_key
                    st.success("✅ API key stored for this session")
                api_key = manual_api_key
            elif 'api_key' in st.session_state:
                api_key = st.session_state.api_key
                st.info("ℹ️ Using stored API key from session")
        
        # Clear stored API key button
        if 'api_key' in st.session_state:
            if st.button("🗑️ Clear stored API key"):
                del st.session_state.api_key
                st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("### 🔑 API Key Setup Methods")
        st.markdown("""
        **Method 1:** Enter key manually above ⬆️  
        **Method 2:** Create a `.env` file with:
        ```
        TOGETHER_API_KEY=your_api_key_here
        ```
        **Method 3:** Set environment variable:
        ```bash
        export TOGETHER_API_KEY=your_key
        ```
        """)
    
    # API Key status indicator
    st.markdown("---")
    if api_key:
        st.markdown("**Status:** 🟢 API Key Ready")
    else:
        st.markdown("**Status:** 🔴 API Key Required")
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image containing text to extract"
        )
        
        if uploaded_file is not None and api_key:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Perform OCR
            with st.spinner("Extracting text..."):
                extracted_text, error = extract_text_with_llama(image, api_key)
            
            if error:
                st.error(f"❌ {error}")
                st.markdown("- Check your API key")
                st.markdown("- Verify you have sufficient API credits")
                st.markdown("- Try uploading a different image")
            else:
                st.success("✅ Text extracted successfully!")
                
                # Display extracted text
                st.text_area(
                    "Extracted Text:",
                    value=extracted_text,
                    height=400,
                    help="You can copy this text by selecting it"
                )
                
                # Text statistics
                word_count = len(extracted_text.split())
                char_count = len(extracted_text)
                st.markdown(f"**Word count:** {word_count}  \n**Character count:** {char_count}")
                
                # --- LangChain Q&A Section ---
                st.markdown("---")
                st.header("❓ Ask a Question")
                st.markdown("Enter a question about the text above, and the model will answer using LangChain retrieval.")
                
                # Prepare the LangChain objects only once per uploaded image
                if extracted_text:
                    # Convert the extracted text into a LangChain Document
                    doc = Document(page_content=extracted_text)
                    
                    # Optionally, split into smaller chunks for long texts
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    docs = text_splitter.split_text(extracted_text)
                    langchain_docs = [Document(page_content=chunk) for chunk in docs]
                    
                    # Initialize the LLM with Together AI endpoint via ChatOpenAI wrapper
                    llm = ChatOpenAI(
                        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                        openai_api_key=api_key,
                        openai_api_base="https://api.together.xyz/v1",
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    # Load a QA chain (stuffing all chunks together)
                    qa_chain = load_qa_chain(llm, chain_type="stuff")
                    
                    question = st.text_input(
                        "Your question:",
                        placeholder="e.g., What is the main instruction given in the image?",
                        key="qa_question"
                    )
                    
                    if question:
                        with st.spinner("Generating answer..."):
                            # Run the QA chain over the documents
                            answer = qa_chain.run(input_documents=langchain_docs, question=question)
                        st.markdown("**Answer:**")
                        st.write(answer)
                else:
                    st.info("No text to create documents from.")
        
        elif uploaded_file and not api_key:
            st.warning("⚠️ Please configure your API key in the sidebar to extract text.")
        
        else:
            st.info("👆 Please upload an image to extract text from it.")
            
            # Sample usage guide
            with st.expander("💡 Usage Tips"):
                st.markdown("""
                **For best results:**
                - Use high-resolution images  
                - Ensure text is clearly visible  
                - Avoid heavily distorted or rotated text  
                - Good contrast between text and background works best  
                
                **Supported image types:** PNG, JPG, JPEG, WEBP  
                """)
    
    with col2:
        st.header("📖 How It Works")
        st.markdown("""
        1. **Upload an image**: The image is sent to Together AI's Llama-3.2-90B-Vision-Instruct-Turbo model to extract text.  
        2. **View extracted text**: You can inspect the text output in the text area.  
        3. **Ask questions**: LangChain splits the extracted text into chunks and uses the same Llama model via Together AI to answer your questions.  
        
        🔗 **Links & References**  
        - [LangChain Documentation](https://langchain.readthedocs.io/)  
        - [Together AI API Reference](https://docs.together.ai/)  
        """)
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("This app uses **Llama-3.2-90B-Vision-Instruct-Turbo** via Together AI for both OCR and Q&A.")
        st.markdown("**Supported formats:** PNG, JPG, JPEG, WEBP")
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>🤖 Powered by <strong>Llama-3.2-90B-Vision-Instruct-Turbo</strong> via Together AI</p>
                <p>Built with ❤️ using Streamlit and LangChain</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()