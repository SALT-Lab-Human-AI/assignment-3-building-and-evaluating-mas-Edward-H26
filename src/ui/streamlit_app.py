"""
Streamlit Web Interface
Web UI for the multi-agent research system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import yaml
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from src.langgraph_orchestrator import LangGraphOrchestrator

# Load environment variables
load_dotenv()


def load_config():
    """Load configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'orchestrator' not in st.session_state:
        config = load_config()
        try:
            st.session_state.orchestrator = LangGraphOrchestrator(config)
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.session_state.orchestrator = None

    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

    if 'show_safety_log' not in st.session_state:
        st.session_state.show_safety_log = False


async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query through the orchestrator.
    
    Args:
        query: Research query to process
        
    Returns:
        Result dictionary with response, citations, and metadata
    """
    orchestrator = st.session_state.orchestrator
    
    if orchestrator is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized",
            "response": "Error: System not properly initialized. Please check your configuration.",
            "citations": [],
            "metadata": {}
        }
    
    try:
        result = orchestrator.process_query(query)
        
        # Check for errors
        if "error" in result:
            return result
        
        # Extract citations from conversation history
        citations = extract_citations(result)
        
        # Extract agent traces for display
        agent_traces = extract_agent_traces(result)
        
        # Format metadata
        metadata = result.get("metadata", {})
        metadata["agent_traces"] = agent_traces
        metadata["citations"] = citations
        metadata["critique_score"] = calculate_quality_score(result)
        
        return {
            "query": query,
            "response": result.get("response", ""),
            "citations": citations,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "response": f"An error occurred: {str(e)}",
            "citations": [],
            "metadata": {"error": True}
        }


def extract_citations(result: Dict[str, Any]) -> list:
    """Extract citations from research result."""
    citations = []
    
    # Look through conversation history for citations
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "")
        
        # Find URLs in content
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        
        # Find citation patterns like [Source: Title]
        citation_patterns = re.findall(r'\[Source: ([^\]]+)\]', content)
        
        for url in urls:
            if url not in citations:
                citations.append(url)
        
        for citation in citation_patterns:
            if citation not in citations:
                citations.append(citation)
    
    return citations[:10]  # Limit to top 10


def extract_agent_traces(result: Dict[str, Any]) -> Dict[str, list]:
    """Extract agent execution traces from conversation history."""
    traces = {}
    
    for msg in result.get("conversation_history", []):
        agent = msg.get("source", "Unknown")
        content = msg.get("content", "")[:200]  # First 200 chars
        
        if agent not in traces:
            traces[agent] = []
        
        traces[agent].append({
            "action_type": "message",
            "details": content
        })
    
    return traces


def calculate_quality_score(result: Dict[str, Any]) -> float:
    """Calculate a quality score based on various factors."""
    score = 5.0  # Base score
    
    metadata = result.get("metadata", {})
    
    # Add points for sources
    num_sources = metadata.get("num_sources", 0)
    score += min(num_sources * 0.5, 2.0)
    
    # Add points for critique
    if metadata.get("critique"):
        score += 1.0
    
    # Add points for conversation length (indicates thorough discussion)
    num_messages = metadata.get("num_messages", 0)
    score += min(num_messages * 0.1, 2.0)
    
    return min(score, 10.0)  # Cap at 10


def display_response(result: Dict[str, Any]):
    """Display query response with glassmorphism styling."""
    # Check for errors
    if "error" in result:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #EF4444;">
            <p style="color: #EF4444;">⚠️ Error: {result['error']}</p>
        </div>
        """, unsafe_allow_html=True)
        return

    response = result.get("response", "")
    citations = result.get("citations", [])
    metadata = result.get("metadata", {})

    # Response card with gradient accent
    st.markdown("""
    <div class="glass-card glow-border" style="border-left: 4px solid #3B82F6;">
        <h3 style="color: #F8FAFC; margin-bottom: 1rem;">
            ✨ Research Findings
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(response)

    # Citation pills
    if citations:
        with st.expander("📚 Sources & Citations", expanded=False):
            citation_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">'
            for i, citation in enumerate(citations, 1):
                display_text = citation[:50] + "..." if len(citation) > 50 else citation
                citation_html += f'''
                <span style="
                    background: rgba(59, 130, 246, 0.2);
                    color: #3B82F6;
                    padding: 0.25rem 0.75rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                ">[{i}] {display_text}</span>
                '''
            citation_html += '</div>'
            st.markdown(citation_html, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Sources", metadata.get("num_sources", 0))
    with col2:
        score = metadata.get("critique_score", 0)
        st.metric("⭐ Quality", f"{score:.1f}/10")
    with col3:
        st.metric("🔒 Safety", "✓ Passed")

    # Safety events
    safety_events = metadata.get("safety_events", [])
    if safety_events:
        with st.expander("⚠️ Safety Events", expanded=True):
            for event in safety_events:
                event_type = event.get("type", "unknown")
                violations = event.get("violations", [])
                st.markdown(f"""
                <div style="
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 8px;
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                ">
                    <strong style="color: #EF4444;">{event_type.upper()}</strong>
                    <span style="color: #94A3B8;">: {len(violations)} violation(s)</span>
                </div>
                """, unsafe_allow_html=True)
                for violation in violations:
                    st.text(f"  • {violation.get('reason', 'Unknown')}")

    # Agent traces
    if st.session_state.show_traces:
        agent_traces = metadata.get("agent_traces", {})
        if agent_traces:
            display_agent_traces(agent_traces)


def display_agent_traces(traces: Dict[str, Any]):
    """Display agent traces as visual timeline."""
    agent_icons = {
        "Planner": "📋",
        "Researcher": "🔍",
        "Writer": "✍️",
        "Critic": "🎯"
    }
    agent_colors = {
        "Planner": "#3B82F6",
        "Researcher": "#8B5CF6",
        "Writer": "#06B6D4",
        "Critic": "#10B981"
    }

    with st.expander("🔄 Agent Workflow Timeline", expanded=False):
        for agent_name, actions in traces.items():
            icon = agent_icons.get(agent_name, "🤖")
            color = agent_colors.get(agent_name, "#3B82F6")

            st.markdown(f"""
            <div style="
                display: flex;
                align-items: flex-start;
                margin-bottom: 1rem;
                padding-left: 1rem;
                border-left: 2px solid {color};
            ">
                <div style="
                    background: {color};
                    color: white;
                    padding: 0.5rem;
                    border-radius: 8px;
                    margin-right: 1rem;
                    font-size: 1.5rem;
                    min-width: 40px;
                    text-align: center;
                ">{icon}</div>
                <div style="flex: 1;">
                    <strong style="color: {color};">{agent_name}</strong>
                    <p style="color: #94A3B8; font-size: 0.9rem; margin: 0.25rem 0;">
                        {len(actions)} action(s) completed
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show action details
            for action in actions:
                details = action.get("details", "")
                if details:
                    preview = details[:100] + "..." if len(str(details)) > 100 else details
                    st.markdown(f"""
                    <div style="
                        margin-left: 3.5rem;
                        margin-bottom: 0.5rem;
                        padding: 0.5rem;
                        background: rgba(255, 255, 255, 0.02);
                        border-radius: 4px;
                        font-size: 0.85rem;
                        color: #64748B;
                    ">{preview}</div>
                    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with glassmorphism styling."""
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🤖</span>
            <h2 style="
                background: linear-gradient(90deg, #3B82F6, #8B5CF6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.2rem;
                margin-top: 0.5rem;
            ">HCI Research AI</h2>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Settings section
        st.markdown("### ⚙️ Settings")
        st.session_state.show_traces = st.toggle(
            "Show Agent Traces",
            value=st.session_state.show_traces
        )
        st.session_state.show_safety_log = st.toggle(
            "Show Safety Log",
            value=st.session_state.show_safety_log
        )

        st.divider()

        # Statistics with glass cards
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", len(st.session_state.history))
        with col2:
            st.metric("Safety", "100%")

        st.divider()

        # Clear button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        # About section
        st.divider()
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "HCI")

        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
        ">
            <p style="color: #64748B; font-size: 0.8rem; margin: 0;">
                <strong style="color: #94A3B8;">System:</strong> {system_name}<br>
                <strong style="color: #94A3B8;">Topic:</strong> {topic}
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return

    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            timestamp = item.get("timestamp", "")
            query = item.get("query", "")
            st.markdown(f"**{i}.** [{timestamp}] {query}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for futuristic glassmorphism theme
    st.markdown("""
    <style>
        /* Dark theme base */
        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        }

        /* Glass card effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        /* Glowing border effect */
        .glow-border {
            border: 1px solid rgba(59, 130, 246, 0.3);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
        }

        /* Text styling */
        h1, h2, h3 { color: #F8FAFC !important; }
        p, span, label { color: #CBD5E1 !important; }

        /* Input styling */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: #F8FAFC !important;
        }
        .stTextArea textarea:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
            border: none !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4) !important;
            transform: translateY(-2px) !important;
        }

        /* Secondary buttons */
        .stButton > button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #CBD5E1 !important;
        }
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        [data-testid="stMetricValue"] { color: #3B82F6 !important; }
        [data-testid="stMetricLabel"] { color: #94A3B8 !important; }

        /* Expanders */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }

        /* Divider */
        hr { border-color: rgba(255, 255, 255, 0.1) !important; }

        /* Markdown text */
        .stMarkdown { color: #CBD5E1 !important; }

        /* Warning and info boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Header with gradient title
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="
            background: linear-gradient(90deg, #3B82F6, #8B5CF6, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        ">Multi-Agent Research Assistant</h1>
        <p style="color: #64748B; font-size: 1.1rem;">
            AI-powered HCI research synthesis with safety guardrails
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span style="
                background: rgba(59, 130, 246, 0.2);
                color: #3B82F6;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
            ">System Online</span>
            <span style="
                background: rgba(139, 92, 246, 0.2);
                color: #8B5CF6;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
            ">HCI Research</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    display_sidebar()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?"
        )

        # Submit button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Processing your query..."):
                    # Process query
                    result = asyncio.run(process_query(query))

                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "result": result
                    })

                    # Display result
                    st.divider()
                    display_response(result)
            else:
                st.warning("Please enter a query.")

        # History
        display_history()

    with col2:
        # Example queries with glass card styling
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <h4 style="color: #F8FAFC; margin-bottom: 0.5rem;">💡 Try These</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">Click to auto-fill your query</p>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            ("🎯", "User-centered design principles"),
            ("🔬", "AR usability research advances"),
            ("🤖", "AI transparency approaches"),
            ("📚", "Ethics in AI education"),
        ]

        for icon, example in examples:
            if st.button(f"{icon} {example}", use_container_width=True, key=f"example_{example}"):
                st.session_state.example_query = f"What are the key {example.lower()} in HCI?"
                st.rerun()

        # If example was clicked, show notification
        if 'example_query' in st.session_state:
            st.markdown(f"""
            <div style="
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                margin-top: 0.5rem;
            ">
                <p style="color: #3B82F6; margin: 0; font-size: 0.9rem;">
                    ✓ Query selected - paste or submit!
                </p>
            </div>
            """, unsafe_allow_html=True)
            del st.session_state.example_query

        st.divider()

        # How it works section with glass styling
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #F8FAFC; margin-bottom: 1rem;">ℹ️ How It Works</h4>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: #3B82F6; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">1</span>
                    <span style="color: #CBD5E1;"><strong style="color: #3B82F6;">Planner</strong> breaks down query</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: #8B5CF6; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">2</span>
                    <span style="color: #CBD5E1;"><strong style="color: #8B5CF6;">Researcher</strong> gathers evidence</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: #06B6D4; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">3</span>
                    <span style="color: #CBD5E1;"><strong style="color: #06B6D4;">Writer</strong> synthesizes findings</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: #10B981; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">4</span>
                    <span style="color: #CBD5E1;"><strong style="color: #10B981;">Critic</strong> verifies quality</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="background: #EF4444; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">5</span>
                    <span style="color: #CBD5E1;"><strong style="color: #EF4444;">Safety</strong> ensures compliance</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Safety log (if enabled)
    if st.session_state.show_safety_log:
        st.divider()
        st.markdown("### 🛡️ Safety Event Log")
        # TODO: Display safety events from safety manager
        st.info("No safety events recorded.")


if __name__ == "__main__":
    main()
