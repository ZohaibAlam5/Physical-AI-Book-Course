"""
Hugging Face Space App for RAG Chatbot
This file creates a Gradio interface for the RAG chatbot
"""
import os
import gradio as gr
import asyncio
from src.api.v1.chat import rag_service
from src.models.query import QueryRequest

# Set environment variables for the services
os.environ.setdefault('QDRANT_URL', os.getenv('QDRANT_URL', 'http://localhost:6333'))
os.environ.setdefault('QDRANT_COLLECTION_NAME', os.getenv('QDRANT_COLLECTION_NAME', 'book_content_chunks'))

def chat_with_rag(message, history):
    """
    Chat function that uses the RAG service to answer questions
    """
    try:
        # Create a query request
        query_request = QueryRequest(
            question=message,
            query_type="global"  # Default to global search
        )

        # Process the query using the RAG service
        response = rag_service.process_query(query_request)

        return response.answer
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Physical AI & Humanoid Robotics Assistant") as demo:
    gr.Markdown("# Physical AI & Humanoid Robotics Assistant")
    gr.Markdown("Ask me anything about the Physical AI & Humanoid Robotics book content!")

    chatbot = gr.Chatbot(
        label="Chat with the Book Assistant",
        bubble_full_width=False,
        avatar_images=(
            "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",  # User avatar
            "https://cdn-icons-png.flaticon.com/512/4712/4712139.png"   # Bot avatar
        )
    )

    msg = gr.Textbox(label="Your Question", placeholder="Ask a question about the book...")
    clear = gr.Button("Clear Chat")

    def respond(message, chat_history):
        bot_message = chat_with_rag(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# For Hugging Face Spaces, we need to define the app as a Gradio interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv('PORT', 7860)),
        share=False,  # Set to True if you want a public link during development
        show_error=True
    )
    