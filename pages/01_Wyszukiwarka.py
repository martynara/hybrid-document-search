import os
os.environ["STREAMLIT_WATCHER_WARNING_DISABLED"] = "true"

import asyncio
import streamlit as st
import time
from asyncio import new_event_loop, set_event_loop

from _style import apply_custom_css
from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Application.Services.Search.SearchService import SearchService, SearchRow, SearchResults
from Application.Services.Search.AdvancedSearchService import AdvancedSearchService
from Application.InternalServices.LLMService import LLMService

# ============== STYLES ==============

def load_styles():
    """
    Load all CSS styles for the application
    """
    st.markdown("""
    <style>
        /* Overall layout improvements */
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }
        
        /* Header styling */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        /* Target the header specifically */
        h1, .stMarkdown h1 {
            background-color: #003366 !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 10px !important;
            text-align: center !important;
            margin-bottom: 30px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        /* Improve form layout */
        .stTextInput, .stSlider {
            margin-bottom: 20px !important;
        }
        
        /* Search button styling */
        .stButton > button {
            background-color: #003366 !important;
            color: white !important;
            padding: 0.5rem 2rem !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            margin: 20px 0 !important;
            display: block !important;
        }
        
        .stButton > button:hover {
            background-color: #004080 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        }
        
        /* Better layout for sections */
        .result-area {
            margin-top: 30px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 10px;
        }
        
        .search-header {
            background-color: #003366;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #FF4B4B;
            text-align: center;
        }
        
        .metadata-item {
            background-color: #eef2f5;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .keyword-badge {
            background-color: #e0e0e0;
            padding: 4px 10px;
            margin: 3px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.9em;
        }
        
        .qa-item {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Improve form elements */
        .stTooltip {
            margin-top: 8px !important;
            margin-bottom: 8px !important;
        }
        
        .stRadio > div {
            display: flex !important;
            flex-direction: row !important;
            gap: 20px !important;
        }
        
        .filter-option {
            margin-bottom: 15px !important;
            padding: 10px !important;
            background-color: #f9f9f9 !important;
            border-radius: 8px !important;
        }
        
        /* Center the toggle and checkbox elements */
        .st-emotion-cache-1r6slb0 {
            justify-content: center !important;
        }
        
        /* Improve column layout */
        [data-testid="column"] {
            padding: 10px !important;
            background-color: #f9f9f9 !important;
            border-radius: 8px !important;
            margin: 5px !important;
        }
        
        /* Results display styling */
        .vector-item {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #003366;
        }
        
        .vector-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .vector-score {
            padding: 5px 10px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }
        
        .vector-area {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 8px;
        }
        
        .answer-container {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# ============== SESSION STATE MANAGEMENT ==============

def initialize_session_state():
    """
    Initialize all session state variables needed for the application
    """
    if 'qa' not in st.session_state:
        st.session_state.qa = []
        
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = "Wpisz pytanie i kliknij 'Szukaj', aby otrzymać odpowiedź."
    
    if 'services_initialized' not in st.session_state:
        st.session_state.services_initialized = False
        st.session_state.qdrant_service = None
        st.session_state.embedding_service = None
        st.session_state.llm_service = None
        st.session_state.startup_times = {}
    
    if 'results' not in st.session_state:
        st.session_state.results = None
        
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
        
    if 'last_search_time' not in st.session_state:
        st.session_state.last_search_time = 0
        
    if 'answer_source' not in st.session_state:
        st.session_state.answer_source = ""

# ============== SERVICE MANAGEMENT ==============

def initialize_services():
    """
    Initializes required services for the search application with performance tracking.
    
    Initializes QdrantManagerService for vector storage, EmbeddingsService for text embeddings,
    SearchService for performing searches, and LLMService for generating answers.
    
    Returns:
        tuple: Contains initialized search_service, embedding_service, and llm_service objects
    """
    if not st.session_state.services_initialized:
        with st.spinner('Trwa inicjalizacja serwisów... To może chwilę potrwać przy pierwszym uruchomieniu'):
            services = {}
            
            # Initialize QDrant service
            start_time = time.time()
            st.session_state.qdrant_service = QdrantManagerService(collection_name="documents", vector_size=384)
            services['qdrant'] = time.time() - start_time
            
            # Initialize Embedding service
            start_time = time.time()
            st.session_state.embedding_service = EmbeddingsService(model_name="all-MiniLM-L6-v2")
            services['embedding'] = time.time() - start_time
            
            # Initialize Search service
            start_time = time.time()
            st.session_state.search_service = SearchService(
                qdrant_service=st.session_state.qdrant_service,
                embedding_service=st.session_state.embedding_service
            )
            services['search'] = time.time() - start_time
            
            # Initialize Advanced Search service
            start_time = time.time()
            st.session_state.advanced_search_service = AdvancedSearchService(
                qdrant_service=st.session_state.qdrant_service,
                search_service=st.session_state.search_service,
                embedding_service=st.session_state.embedding_service
            )
            services['advanced_search'] = time.time() - start_time
            
            # Initialize LLM service
            start_time = time.time()
            try:
                st.session_state.llm_service = LLMService.create_openai()
                services['llm'] = time.time() - start_time
            except Exception as e:
                st.error(f"Nie udało się zainicjalizować serwisu LLM: {str(e)}")
                st.error("Upewnij się, że klucz API OpenAI jest poprawnie skonfigurowany w pliku .env")
                st.session_state.llm_service = None
            
            # Update session state
            st.session_state.services_initialized = True
            st.session_state.startup_times = services
            
            # Log timing information
            total_time = sum(services.values())
            print(f"Total services initialization: {total_time:.2f} seconds")
            for service, duration in services.items():
                print(f"{service.capitalize()} service initialization: {duration:.2f} seconds")
    
    return (
        st.session_state.search_service, 
        st.session_state.embedding_service, 
        st.session_state.llm_service
    )

# ============== ASYNCIO UTILITIES ==============

def run_async(coroutine):
    """
    Safely runs an asyncio coroutine in a Streamlit context.
    
    Creates a new event loop if one doesn't exist in the current thread,
    runs the coroutine to completion, and cleans up the loop when done.
    
    Args:
        coroutine: An asyncio coroutine to execute
        
    Returns:
        The result of the coroutine execution
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = new_event_loop()
        set_event_loop(loop)
    
    return loop.run_until_complete(coroutine)

# ============== SEARCH FUNCTIONALITY ==============

async def search_hybrid(query, search_service, embedding_service, limit=5, search_type="NLP", 
                       metadata_filter=None, keywords=None, summary_contains=None, qa_contains=None):
    """
    Performs search across documents using the specified search type and filters.
    
    Args:
        query: The search query text
        search_service: Service for performing searches
        embedding_service: Service for creating text embeddings
        limit: Maximum number of results to return
        search_type: "NLP" for semantic search or other values for contextual search
        metadata_filter: Optional filter for document metadata
        keywords: Keywords to filter search results
        summary_contains: Filter for text in document summaries
        qa_contains: Filter for text in QA pairs
        
    Returns:
        SearchResults object containing search results or empty SearchResults on error
    """
    try:
        search_params = {
            "query": query,
            "limit": limit
        }
        
        if search_type == "NLP":
            results = await search_service.search_semantic(**search_params)
        else:
            if metadata_filter:
                search_params["metadata"] = metadata_filter
            
            if keywords:
                search_params["keywords"] = keywords
                
            if summary_contains:
                search_params["summary_contains"] = summary_contains
                
            if qa_contains:
                search_params["qa_contains"] = qa_contains
                
            results = await search_service.advanced_search(**search_params)
        
        return results
    except Exception as e:
        st.error(f"Błąd wyszukiwania: {str(e)}")
        return SearchResults()

# ============== ANSWER GENERATION ==============

def generate_answer_with_llm(query, search_results, llm_service):
    """
    Generate an answer using LLM based on search results.
    
    Creates a prompt using context from search results and uses LLM to generate
    a natural language answer to the user's query.
    
    Args:
        query: User's original query
        search_results: SearchResults object with found documents
        llm_service: Service for generating LLM answers
        
    Returns:
        str: Generated answer or error message
    """
    if not llm_service:
        return "Serwis LLM nie jest dostępny. Sprawdź konfigurację API OpenAI."
    
    if not search_results or search_results.count == 0:
        return "Nie znaleziono odpowiednich informacji dla tego zapytania."
    
    # Build context from search results
    context_parts = []
    for i, row in enumerate(search_results.rows, 1):
        text = row.text
        metadata = row.metadata or {}
        summary = metadata.get('summary', '')
        
        if summary:
            context_parts.append(f"Źródło {i}: {summary}")
        else:
            snippet = text[:1000] + "..." if len(text) > 1000 else text
            context_parts.append(f"Źródło {i}: {snippet}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for LLM
    prompt = f"""Jako asystent, odpowiedz na pytanie użytkownika na podstawie podanych źródeł informacji.
Używaj tylko informacji z podanych źródeł. Jeśli nie ma wystarczających informacji, 
powiedz, że nie masz wystarczających danych, aby udzielić dokładnej odpowiedzi.

Pytanie użytkownika: {query}

Źródła informacji:
{context}

Twoja odpowiedź:"""
    
    try:
        # Run the async complete method using run_async
        answer = run_async(llm_service.complete(prompt))
        return answer
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Błąd podczas generowania odpowiedzi: {str(e)}")
        st.error(f"Szczegóły błędu: {error_details}")
        return f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}. Spróbuj ponownie później."

def update_answer(search_results=None, query=None, use_llm=True):
    """
    Updates the current answer based on search results.
    
    Uses either LLM generation or retrieval from existing QA pairs to update
    the answer displayed to the user.
    
    Args:
        search_results: SearchResults with found documents
        query: Original user query
        use_llm: Whether to use LLM to generate answers
    """
    if not search_results or search_results.count == 0:
        st.session_state.current_answer = "Nie znaleziono odpowiedzi na to pytanie."
        st.session_state.answer_source = "retrieval"
        return
    
    # Try to generate answer using LLM if enabled
    if use_llm and st.session_state.llm_service and query:
        generated_answer = generate_answer_with_llm(query, search_results, st.session_state.llm_service)
        st.session_state.current_answer = generated_answer
        st.session_state.answer_source = "generated"
        return
    
    # Otherwise, try to find answer from QA pairs
    best_answer = None
    qa_pairs = []
    
    # Extract QA pairs from search results
    for row in search_results.rows:
        metadata = row.metadata or {}
        
        if 'qa_pairs' in metadata and metadata['qa_pairs']:
            qa_pairs.extend(metadata['qa_pairs'])
        
        if 'global_qa_pairs' in metadata and metadata['global_qa_pairs']:
            qa_pairs.extend(metadata['global_qa_pairs'])
    
    # Use first QA pair as answer if available
    if qa_pairs and len(qa_pairs) > 0:
        st.session_state.qa = qa_pairs
        
        first_qa = qa_pairs[0]
        best_answer = first_qa.get('answer', first_qa.get('response', ''))
    
    # Fallback to document text if no QA pairs
    if not best_answer and search_results.count > 0:
        best_answer = search_results.rows[0].text
        if len(best_answer) > 500:
            best_answer = best_answer[:500] + "..."
    
    st.session_state.current_answer = best_answer or "Nie znaleziono odpowiedzi na to pytanie."
    st.session_state.answer_source = "retrieval"

# ============== UI COMPONENTS ==============

def render_header():
    """Render the application header"""
    st.markdown("<h1>Wyszukiwarka</h1>", unsafe_allow_html=True)

def render_search_form():
    """
    Render the search form with all inputs and options
    
    Returns:
        tuple: Contains all form values needed for search
    """
    # Main search query
    query = st.text_input("Wpisz zapytanie:", "Gdzie mogą być składane oświadczenia związane z prowadzoną likwidacją szkody?")
    
    # Create a better layout for options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main search options
        limit = st.slider("Liczba wyników:", 1, 10, 5)
        use_llm = st.toggle("Generuj odpowiedzi za pomocą modelu AI", value=True, 
                          help="Włącz, aby generować odpowiedzi za pomocą modelu AI zamiast odzyskiwać wcześniej zdefiniowane odpowiedzi")
        
    with col2:
        # Advanced options in a container
        with st.container():
            st.markdown("<div style='background-color:#f5f5f5; padding:15px; border-radius:10px;'>", unsafe_allow_html=True)
            st.markdown("### Multiple Collections Search")
            use_keywords_collection = st.checkbox("Słowa kluczowe", value=False,
                                              help="Wyszukiwanie w kolekcji słów kluczowych, +1 do scoringu za dopasowanie")
            use_summaries_collection = st.checkbox("Streszczenia", value=False,
                                               help="Wyszukiwanie w kolekcji streszczeń, +1 do scoringu za dopasowanie")
            debug_mode = st.checkbox("Tryb debugowania", value=False,
                                 help="Pokaż szczegółowe informacje o wynikach z dodatkowych kolekcji")
            st.markdown("</div>", unsafe_allow_html=True)
            
    return query, limit, use_llm, use_keywords_collection, use_summaries_collection, debug_mode

def render_search_button():
    """
    Render a centered search button
    
    Returns:
        bool: True if button was clicked, False otherwise
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        return st.button("Szukaj", key="search_button")

def render_answer_section(answer, answer_source, search_results):
    """
    Render the answer section with the current answer and related QA pairs
    
    Args:
        answer: The answer text to display
        answer_source: Source of the answer ('generated' or 'retrieval')
        search_results: SearchResults with found documents
    """
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    st.subheader("Odpowiedź")
    st.markdown('<div class="section-container">', unsafe_allow_html=True)

    if answer:
        st.markdown(f"""
        <div class="answer-container">
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Display answer source info
        if answer_source == "generated":
            st.info("Ta odpowiedź została wygenerowana przez model AI na podstawie znalezionych dokumentów.")
        else:
            st.info("Ta odpowiedź została odzyskana z bazy danych (predefiniowany QA).")
        
        # Display related QA pairs if available
        if search_results and search_results.count > 0:
            render_related_qa_pairs(search_results)
    else:
        st.markdown("""
        <div class="answer-container" style="color: #666;">
            Odpowiedź pojawi się tutaj po zadaniu pytania.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_related_qa_pairs(search_results):
    """
    Render related QA pairs from search results
    
    Args:
        search_results: SearchResults with found documents
    """
    st.subheader("Powiązane pytania i odpowiedzi:")
    
    # Extract QA pairs from top 3 results
    top_qa_pairs = []
    for i, row in enumerate(search_results.rows[:3], 1):
        metadata = row.metadata or {}
        
        # Get QA pairs from metadata
        qa_pairs = metadata.get('qa_pairs', [])
        
        # Also get global QA pairs if available
        if 'global_qa_pairs' in metadata:
            qa_pairs.extend(metadata.get('global_qa_pairs', []))
        
        # Add source information to each QA pair
        for qa in qa_pairs:
            qa['source'] = i
            qa['source_title'] = metadata.get('source', f'Źródło {i}')
        
        top_qa_pairs.extend(qa_pairs)
    
    # Display top 3 QA pairs in expandable sections
    if top_qa_pairs:
        for i, qa in enumerate(top_qa_pairs[:3], 1):
            question = qa.get('question', qa.get('query', f"Pytanie {i}"))
            answer = qa.get('answer', qa.get('response', "Brak odpowiedzi"))
            source = qa.get('source', '')
            source_title = qa.get('source_title', f'Źródło: {source}')
            
            with st.expander(f"Q: {question}"):
                st.markdown(f"**A:** {answer}")
                st.caption(f"Źródło: {source_title}")
    else:
        st.info("Brak powiązanych pytań i odpowiedzi dla tego zapytania.")

def render_search_results(results):
    """
    Render the search results section
    
    Args:
        results: SearchResults with found documents
    """
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    st.subheader("Wyszukane dokumenty")
    st.markdown('<div class="section-container">', unsafe_allow_html=True)

    for i, row in enumerate(results.rows):
        render_search_result_item(row, i)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_search_result_item(row, index):
    """
    Render a single search result item
    
    Args:
        row: SearchRow with document data
        index: Result index for numbering
    """
    metadata = row.metadata or {}
    
    # Extract scoring details
    base_score = metadata.get('base_score', 0)
    bonus_score = metadata.get('bonus_score', 0)
    matched_collections = metadata.get('matched_collections', ["documents"])
    
    # Format score display - just show the final score
    score_details = f"{row.score:.2f}"
    
    # Determine score color based on value
    score_color = "#2ecc71"  # green
    if row.score < 0.7:
        score_color = "orange"
    if row.score < 0.5:
        score_color = "red"
    
    # Get document title
    title = 'Brak tytułu'
    if 'source' in metadata:
        title = metadata['source']
    elif 'file_name' in metadata:
        title = metadata['file_name']
    
    # Format matched collections for display
    collection_names = []
    for c in matched_collections:
        if c == "documents":
            collection_names.append("Główna")
        elif c == "documents_keywords":
            collection_names.append("Słowa kluczowe")
        elif c == "documents_summaries":
            collection_names.append("Streszczenia")
        else:
            collection_names.append(c.replace('documents_', ''))
    
    # Render result item
    st.markdown(f"""
    <div class="vector-item">
        <div class="vector-header">
            <h3>{title}</h3>
            <div class="vector-score" style="background-color:{score_color};">{score_details}</div>
        </div>
        <div class="vector-area">
        {metadata.get('document_type', 'Dokument')} • 
        Znaleziono w: {', '.join(collection_names)}
        </div>
        <div style="margin-top: 10px;">{row.text[:300]}...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render metadata sections if available
    render_result_metadata(row, metadata)

def render_result_metadata(row, metadata):
    """
    Render metadata sections for a search result
    
    Args:
        row: SearchRow with document data
        metadata: Document metadata dictionary
    """
    # Summary
    summary = metadata.get('summary', '')
    if summary:
        st.markdown("**Podsumowanie:**")
        st.info(summary)
    
    # Keywords
    keywords = metadata.get('keywords', [])
    if keywords:
        st.markdown("**Słowa kluczowe:**")
        keyword_html = " ".join([f"<span class='keyword-badge'>{k}</span>" 
                              for k in keywords])
        st.markdown(keyword_html, unsafe_allow_html=True)
    
    # Links
    if row.links:
        st.markdown("**Linki:**")
        for link in row.links:
            st.markdown(f"- [{link}]({link})")
    
    # QA pairs
    qa_pairs = metadata.get('qa_pairs', [])
    if 'global_qa_pairs' in metadata:
        qa_pairs.extend(metadata.get('global_qa_pairs', []))
    
    if qa_pairs and len(qa_pairs) > 0:
        with st.expander(f"Pytania i odpowiedzi dla tego fragmentu ({len(qa_pairs)})"):
            for j, qa in enumerate(qa_pairs):
                question = qa.get('question', qa.get('query', f"Pytanie {j+1}"))
                answer = qa.get('answer', qa.get('response', "Brak odpowiedzi"))
                
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
                st.markdown("---")

def render_debug_info(debug_mode):
    """
    Render debugging information about the search results
    
    Shows details about search results from each collection when debug mode is enabled
    
    Args:
        debug_mode: Whether debug mode is enabled
    """
    if not debug_mode or 'collection_results' not in st.session_state:
        return
    
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    st.subheader("Informacje debugowania")
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    collection_results = st.session_state.collection_results
    
    for collection, results in collection_results.items():
        if collection == "documents":
            continue  # Skip main collection since it's already shown
            
        with st.expander(f"Wyniki z kolekcji: {collection} ({results.count} znalezionych)"):
            for i, row in enumerate(results.rows):
                metadata = row.metadata or {}
                document_id = metadata.get('document_id', 'brak')
                chunk_id = metadata.get('chunk_id', 'brak')
                score = row.score
                
                st.markdown(f"### Wynik {i+1}")
                st.markdown(f"**Score:** {score:.4f}")
                st.markdown(f"**Document ID:** {document_id}")
                st.markdown(f"**Chunk ID:** {chunk_id}")
                
                if collection == "documents_keywords":
                    keywords = metadata.get('keywords', [])
                    st.markdown(f"**Słowa kluczowe:** {', '.join(keywords)}")
                
                st.markdown("**Treść:**")
                st.text_area("", value=row.text, height=150, key=f"debug_{collection}_{i}")
                st.markdown("---")
    
    # Display matches analysis
    if 'results' in st.session_state and st.session_state.results and st.session_state.results.count > 0:
        with st.expander("Analiza dopasowanych dokumentów między kolekcjami"):
            for i, row in enumerate(st.session_state.results.rows):
                metadata = row.metadata or {}
                matched_collections = metadata.get('matched_collections', [])
                
                if len(matched_collections) > 1:
                    st.markdown(f"### Dokument {i+1}")
                    st.markdown(f"**Znaleziono w kolekcjach:** {', '.join(matched_collections)}")
                    
                    if 'matched_rows' in metadata:
                        matched_rows = metadata.get('matched_rows', {})
                        
                        for coll, matched_row in matched_rows.items():
                            if coll != "documents":
                                st.markdown(f"**Treść z kolekcji {coll}:**")
                                st.text_area("", value=matched_row.text, height=100, key=f"match_{i}_{coll}")
                    
                    st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============== MAIN APPLICATION ==============

def perform_search(query, limit, collections):
    """
    Perform search operation with error handling and timing
    
    Args:
        query: User's search query
        limit: Maximum number of results to return
        collections: List of collections to search in
        
    Returns:
        SearchResults with search results or None on error
    """
    try:
        start_time = time.time()
        
        advanced_search_service = st.session_state.advanced_search_service
        
        results = run_async(advanced_search_service.multi_collection_search(
            query=query,
            collections=collections,
            limit=limit
        ))
        
        # Record search time
        search_time = time.time() - start_time
        st.session_state.last_search_time = search_time
        
        # Display search time
        st.caption(f"Wyszukiwanie zajęło {search_time:.2f} sekund")
        
        return results
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None

def main():
    """
    Main function that runs the Streamlit search application.
    
    Sets up the UI, handles search functionality, and displays results.
    """
    # Initialize styles and state
    load_styles()
    apply_custom_css()
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Initialize services
    search_service, embedding_service, llm_service = initialize_services()
    
    # Render search form
    query, limit, use_llm, use_keywords, use_summaries, debug_mode = render_search_form()
    
    # Render search button
    search_clicked = render_search_button()
    
    # Handle search
    if search_clicked:
        with st.spinner('Trwa wyszukiwanie...'):
            # Prepare collections list
            collections = ["documents"]
            if use_keywords:
                collections.append("documents_keywords")
            if use_summaries:
                collections.append("documents_summaries")
            
            # Perform search
            results = perform_search(query, limit, collections)
            
            if results:
                # Update answer and session state
                update_answer(results, query, use_llm)
                st.session_state.results = results
                st.session_state.last_query = query
        
        # Display search results if available
        if results and results.count > 0:
            st.markdown("<div class='result-area'>", unsafe_allow_html=True)
            
            # Display answer section
            render_answer_section(
                st.session_state.current_answer, 
                st.session_state.answer_source, 
                results
            )
            
            # Display search results
            render_search_results(results)
            
            # Display debug information if debug mode is enabled
            render_debug_info(debug_mode)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Nie znaleziono wyników.")
            
            # Show debug info even if no main results
            render_debug_info(debug_mode)

if __name__ == "__main__":
    main()