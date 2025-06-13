import streamlit as st
from datetime import datetime, timedelta
from app import ParliamentaryQA
from data_processor import SansadAPIClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "qa_system" not in st.session_state:
    st.session_state.qa_system = ParliamentaryQA()
if "api_client" not in st.session_state:
    st.session_state.api_client = SansadAPIClient()

def main():
    st.title("Parliamentary Q&A Assistant")
    
    # Sidebar for ministry selection and data fetching
    with st.sidebar:
        st.header("Settings")
        ministry = st.selectbox(
            "Select Ministry",
            ["Finance", "Defence", "External Affairs", "Home Affairs"]
        )
        
        date_range = st.date_input(
            "Select Date Range",
            value=(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
        )
        
        if st.button("Fetch Latest Data"):
            try:
                with st.spinner("Fetching and processing data..."):
                    raw_data = st.session_state.api_client.fetch_questions(
                        ministry=ministry,
                        date_range={
                            "start": date_range[0].strftime("%Y-%m-%d"),
                            "end": date_range[1].strftime("%Y-%m-%d")
                        }
                    )
                    
                    processed_data = st.session_state.api_client.process_questions(raw_data)
                    st.session_state.qa_system.add_documents(processed_data)
                    st.success("Data updated successfully!")
            except Exception as e:
                st.error(f"Error updating data: {str(e)}")
    
    # Main Q&A interface
    st.header("Ask a Question")
    query = st.text_input("Enter your question about parliamentary proceedings:")
    
    if query:
        try:
            with st.spinner("Searching for relevant information..."):
                # Get relevant documents
                results = st.session_state.qa_system.query_documents(query)
                
                # Generate response
                context = [doc for doc in results["documents"][0]]
                response = st.session_state.qa_system.get_llm_response(query, context)
                
                # Display response
                st.write("Response:")
                st.write(response)
                
                # Display sources
                with st.expander("View Sources"):
                    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                        st.write("---")
                        st.write(f"**Ministry:** {metadata['ministry']}")
                        st.write(f"**Date:** {metadata['date']}")
                        st.write(f"**Member:** {metadata['member']}")
                        st.write(doc)
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()