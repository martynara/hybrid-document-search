import os
os.environ["STREAMLIT_WATCHER_WARNING_DISABLED"] = "true"

import streamlit as st
from _style import *
import pandas as pd
import numpy as np
import asyncio
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime

from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Services.Search.SearchService import SearchService
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Application.Services.QualityMonitor.QualityMonitor import QualityMonitor, QualityReport


st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .result-row {
        background-color: #f0f7ff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .header {
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .subheader {
        padding: 5px;
        background-color: #f0f7ff;
        border-left: 5px solid #4CAF50;
        margin: 20px 0 10px 0;
    }
    .match {
        color: green;
        font-weight: bold;
    }
    .no-match {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

if 'quality_monitor' not in st.session_state:
    st.session_state.quality_monitor = None
    st.session_state.services_initialized = False
    st.session_state.latest_report = None
    st.session_state.test_queries = []
    st.session_state.comparison_results = {}
    st.session_state.performance_data = {
        "timestamps": [],
        "avg_latency_ms": [],
        "query_count": []
    }

def initialize_services():
    if not st.session_state.services_initialized:
        with st.spinner('Initializing services...'):
            try:
                qdrant_service = QdrantManagerService(collection_name="documents", vector_size=384)
                embedding_service = EmbeddingsService(model_name="all-MiniLM-L6-v2")
                search_service = SearchService(qdrant_service=qdrant_service, embedding_service=embedding_service)
                st.session_state.quality_monitor = QualityMonitor(search_service=search_service)
                st.session_state.services_initialized = True
                st.success("Services initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing services: {str(e)}")
                st.error("Make sure QDrant service is running and the collection exists.")

def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

apply_custom_css()
    
display_header("Quality Monitor")

st.sidebar.title("⚙️ Controls")

if not st.session_state.services_initialized:
    if st.sidebar.button("Initialize Services"):
        initialize_services()
else:
    st.sidebar.success("Services are initialized")

page = st.sidebar.selectbox(
    "Select page",
    ["Test Query Management", "Run Evaluation", "Compare Search Strategies", "Performance Monitoring"],
    index=0
)

if st.session_state.services_initialized:
    
    if page == "Test Query Management":
        st.markdown("<div class='subheader'><h2>Test Query Management</h2></div>", unsafe_allow_html=True)
        
        st.subheader("Add Test Query")
        with st.form("add_query_form"):
            query = st.text_input("Query:")
            chunk_id = st.text_input("Expected Chunk ID:")
            
            submit_button = st.form_submit_button("Add Query")
            if submit_button:
                if query and chunk_id:
                    st.session_state.quality_monitor.add_test_query(query, [chunk_id])
                    st.session_state.test_queries = st.session_state.quality_monitor.test_queries.copy()
                    st.success(f"Added test query: '{query}'")
                else:
                    st.error("Both query and chunk ID are required")
        
        st.subheader("Upload queryanswer.json")
        uploaded_file = st.file_uploader("Upload JSON file with test queries", type=["json"])
        if uploaded_file is not None:
            try:
                with open("temp_queryanswer.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.quality_monitor.load_test_queries_from_json("temp_queryanswer.json")
                st.session_state.test_queries = st.session_state.quality_monitor.test_queries.copy()
                st.success(f"Loaded {len(st.session_state.test_queries)} test queries from JSON")
                
            except Exception as e:
                st.error(f"Error loading JSON file: {str(e)}")
                st.error("Make sure the JSON has the correct format with 'query' and 'chunk_id' fields")
        
        st.subheader("Current Test Queries")
        
        if st.session_state.test_queries:
            query_data = []
            for query, expected_chunk_ids in st.session_state.test_queries:
                query_data.append({
                    "Query": query,
                    "Expected Chunk ID": ", ".join(expected_chunk_ids),
                    "Number of Expected Chunks": len(expected_chunk_ids)
                })
            df = pd.DataFrame(query_data)
            st.dataframe(df)