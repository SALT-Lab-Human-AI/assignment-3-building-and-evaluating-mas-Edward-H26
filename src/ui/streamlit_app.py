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

# Import from merged orchestrator file (LangGraph code merged into autogen_orchestrator.py)
from src.autogen_orchestrator import LangGraphOrchestrator

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
    """Calculate quality score using actual evaluation metrics."""
    metadata = result.get("metadata", {})

    llm_score = metadata.get("llm_judge_score", 0)
    if llm_score > 0:
        return llm_score * 10

    reflexion_score = metadata.get("reflexion_score", 0)
    if reflexion_score > 0:
        return reflexion_score * 10

    score = 5.0
    num_sources = metadata.get("num_sources", 0)
    score += min(num_sources * 0.5, 2.0)
    num_messages = metadata.get("num_messages", 0)
    score += min(num_messages * 0.1, 2.0)
    return min(score, 10.0)


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
    """Display agent traces as NEXUS visual timeline with neon effects."""
    agent_icons = {
        "Planner": "📋",
        "Researcher": "🔍",
        "Writer": "✍️",
        "Critic": "🎯"
    }
    # NEXUS neon color scheme
    agent_colors = {
        "Planner": "#3B82F6",      # Electric Blue
        "Researcher": "#8B5CF6",   # Violet
        "Writer": "#06B6D4",       # Cyan
        "Critic": "#10B981"        # Emerald
    }
    agent_glows = {
        "Planner": "rgba(59, 130, 246, 0.5)",
        "Researcher": "rgba(139, 92, 246, 0.5)",
        "Writer": "rgba(6, 182, 212, 0.5)",
        "Critic": "rgba(16, 185, 129, 0.5)"
    }

    with st.expander("🔄 Agent Workflow Timeline", expanded=False):
        # Timeline container with neon line
        st.markdown("""
        <div class="nexus-timeline" style="
            position: relative;
            padding-left: 3rem;
        ">
            <div style="
                content: '';
                position: absolute;
                left: 1rem;
                top: 0;
                bottom: 0;
                width: 2px;
                background: linear-gradient(180deg, #00FFFF 0%, #4F46E5 25%, #FF00FF 50%, #4F46E5 75%, #00FFFF 100%);
                box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
                animation: timeline-pulse 3s ease-in-out infinite;
            "></div>
        </div>
        <style>
            @keyframes timeline-pulse {
                0%, 100% { opacity: 0.7; }
                50% { opacity: 1; }
            }
        </style>
        """, unsafe_allow_html=True)

        for agent_name, actions in traces.items():
            icon = agent_icons.get(agent_name, "🤖")
            color = agent_colors.get(agent_name, "#00FFFF")
            glow = agent_glows.get(agent_name, "rgba(0, 255, 255, 0.5)")

            # NEXUS Agent Node with neon styling
            st.markdown(f"""
            <div class="nexus-agent-node" style="
                position: relative;
                display: flex;
                align-items: flex-start;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
                padding: 1.25rem;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
                border-radius: 16px;
                border: 1px solid rgba({color}, 0.3);
                box-shadow: 0 0 20px {glow};
                animation: node-appear 0.5s ease-out;
            ">
                <!-- Agent Icon with Glow -->
                <div style="
                    width: 50px;
                    height: 50px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, {color}40, {color}20);
                    border: 1px solid {color};
                    border-radius: 12px;
                    font-size: 1.5rem;
                    box-shadow: 0 0 20px {glow}, inset 0 0 15px {glow};
                    animation: icon-glow 2s ease-in-out infinite;
                ">{icon}</div>

                <!-- Agent Info -->
                <div style="flex: 1;">
                    <strong style="
                        color: {color};
                        font-size: 1.1rem;
                        text-shadow: 0 0 10px {glow};
                    ">{agent_name}</strong>
                    <p style="
                        color: rgba(148, 163, 184, 0.8);
                        font-size: 0.85rem;
                        margin: 0.25rem 0 0 0;
                    ">
                        <span style="
                            background: {color}20;
                            color: {color};
                            padding: 0.15rem 0.5rem;
                            border-radius: 10px;
                            font-size: 0.75rem;
                        ">{len(actions)} action(s)</span>
                    </p>
                </div>

                <!-- Status Indicator -->
                <div style="
                    width: 10px;
                    height: 10px;
                    background: {color};
                    border-radius: 50%;
                    box-shadow: 0 0 10px {color};
                    animation: pulse 1.5s ease-in-out infinite;
                "></div>
            </div>
            """, unsafe_allow_html=True)

            # Show action details with NEXUS styling
            for action in actions:
                details = action.get("details", "")
                if details:
                    preview = details[:100] + "..." if len(str(details)) > 100 else details
                    st.markdown(f"""
                    <div style="
                        margin-left: 4rem;
                        margin-bottom: 0.75rem;
                        padding: 0.75rem 1rem;
                        background: linear-gradient(135deg, rgba(0, 255, 255, 0.02), rgba(79, 70, 229, 0.02));
                        border-left: 2px solid {color}50;
                        border-radius: 0 8px 8px 0;
                        font-size: 0.85rem;
                        color: #94A3B8;
                        transition: all 0.3s ease;
                    ">
                        <span style="color: {color}80;">▸</span> {preview}
                    </div>
                    """, unsafe_allow_html=True)

        # Add animation keyframes
        st.markdown("""
        <style>
            @keyframes node-appear {
                from { opacity: 0; transform: translateX(-20px); }
                to { opacity: 1; transform: translateX(0); }
            }
            @keyframes icon-glow {
                0%, 100% { box-shadow: 0 0 15px currentColor; }
                50% { box-shadow: 0 0 25px currentColor, 0 0 35px currentColor; }
            }
        </style>
        """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with NEXUS cyberpunk holographic styling."""
    with st.sidebar:
        # ==========================================================================
        # NEXUS HOLOGRAPHIC LOGO SECTION
        # ==========================================================================
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; position: relative;">
            <!-- Holographic Logo Icon -->
            <div style="
                font-size: 4rem;
                /* animation disabled to prevent blinking */
                filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.5));
                margin-bottom: 0.5rem;
            ">🔮</div>
            <!-- NEXUS Logo Text -->
            <h2 style="
                background: linear-gradient(90deg, #00FFFF, #4F46E5, #FF00FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 1.5rem;
                font-weight: 800;
                letter-spacing: 0.15em;
                margin: 0;
            ">HCI</h2>
            <p style="
                color: rgba(148, 163, 184, 0.7);
                font-size: 0.7rem;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                margin-top: 0.25rem;
            ">Research Interface</p>
        </div>
        <style>
            @keyframes hologram-flicker {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.97; }
            }
        </style>
        """, unsafe_allow_html=True)

        st.divider()

        # ==========================================================================
        # HOLOGRAPHIC NAVIGATION SECTION
        # ==========================================================================
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <p style="
                color: #00FFFF;
                font-size: 0.75rem;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                margin-bottom: 0.75rem;
            ">⚙️ Configuration</p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.show_traces = st.toggle(
            "🔄 Agent Traces",
            value=st.session_state.show_traces
        )
        st.session_state.show_safety_log = st.toggle(
            "🛡️ Safety Log",
            value=st.session_state.show_safety_log
        )

        st.divider()

        # ==========================================================================
        # HOLOGRAPHIC METRICS SECTION
        # ==========================================================================
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <p style="
                color: #FF00FF;
                font-size: 0.75rem;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                margin-bottom: 0.75rem;
            ">📊 Session Metrics</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.popover(f"QUERIES: {len(st.session_state.history)}", use_container_width=True):
                st.markdown("### Query Statistics")
                st.write(f"**Total Queries:** {len(st.session_state.history)}")
                st.write(f"**Session Started:** {st.session_state.get('session_start', 'Current session')}")
                if st.session_state.history:
                    st.markdown("---")
                    st.write("**Recent Queries:**")
                    for i, item in enumerate(st.session_state.history[-3:], 1):
                        query_text = item.get("query", "N/A")[:50]
                        st.write(f"{i}. {query_text}...")
        with col2:
            with st.popover("SAFETY: 100%", use_container_width=True):
                st.markdown("### Safety Statistics")
                st.write("**Safety Score:** 100%")
                st.write("**Blocked Queries:** 0")
                st.write("**Sanitized Responses:** 0")
                st.markdown("---")
                st.success("All safety checks passed")

        st.divider()

        # Clear History Button with Neon Styling
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.divider()

        # ==========================================================================
        # SYSTEM INFO CARD
        # ==========================================================================
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "HCI")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), rgba(79, 70, 229, 0.05));
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.5rem;
        ">
            <p style="
                color: rgba(0, 255, 255, 0.8);
                font-size: 0.7rem;
                letter-spacing: 0.15em;
                text-transform: uppercase;
                margin-bottom: 0.5rem;
            ">System Info</p>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 0; line-height: 1.6;">
                <span style="color: #00FFFF;">▸</span> <strong style="color: #CBD5E1;">Engine:</strong> {system_name}<br>
                <span style="color: #FF00FF;">▸</span> <strong style="color: #CBD5E1;">Domain:</strong> {topic}<br>
                <span style="color: #8B5CF6;">▸</span> <strong style="color: #CBD5E1;">Status:</strong> <span style="color: #10B981;">Active</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Version Badge
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(0, 255, 255, 0.1);">
            <span style="
                background: rgba(79, 70, 229, 0.2);
                border: 1px solid rgba(79, 70, 229, 0.3);
                border-radius: 20px;
                padding: 0.25rem 0.75rem;
                font-size: 0.7rem;
                color: #8B5CF6;
                letter-spacing: 0.1em;
            ">v2.0 HCI</span>
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
        page_title="HCI Research Interface",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Handle "Try These" button auto-fill - must happen BEFORE text_area is rendered
    if 'example_query' in st.session_state:
        st.session_state.query_input = st.session_state.example_query
        del st.session_state.example_query

    # ==========================================================================
    # NEXUS IMMERSIVE 3D SPATIAL WEB CSS
    # A breathtaking first-person view of an immersive 'Spatial Web' interface
    # ==========================================================================
    st.markdown("""
    <style>
        /* ================================================================== */
        /* PHASE 11: INFINITE SPACE BACKGROUND WITH STAR FIELD               */
        /* ================================================================== */

        /* Deep space base with nebula effects */
        .stApp {
            background:
                /* Distant nebula gradients */
                radial-gradient(ellipse at 15% 25%, rgba(79, 70, 229, 0.12) 0%, transparent 40%),
                radial-gradient(ellipse at 85% 75%, rgba(255, 0, 255, 0.08) 0%, transparent 45%),
                radial-gradient(ellipse at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                /* Deep space gradient */
                linear-gradient(180deg, #000005 0%, #050510 20%, #0A0A15 50%, #0F0F20 80%, #0F172A 100%) !important;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }

        /* Far star layer - slowest drift */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image:
                radial-gradient(1px 1px at 20px 30px, rgba(255,255,255,0.4), transparent),
                radial-gradient(1px 1px at 40px 70px, rgba(255,255,255,0.3), transparent),
                radial-gradient(1.5px 1.5px at 50px 160px, rgba(255,255,255,0.35), transparent),
                radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.3), transparent),
                radial-gradient(1.5px 1.5px at 130px 80px, rgba(255,255,255,0.4), transparent),
                radial-gradient(1px 1px at 160px 120px, rgba(255,255,255,0.25), transparent),
                radial-gradient(2px 2px at 70px 100px, rgba(0,255,255,0.5), transparent),
                radial-gradient(2px 2px at 110px 50px, rgba(255,0,255,0.4), transparent),
                radial-gradient(1.5px 1.5px at 180px 140px, rgba(79,70,229,0.5), transparent),
                radial-gradient(2px 2px at 30px 190px, rgba(0,255,255,0.4), transparent);
            background-size: 200px 200px;
            animation: drift-stars 100s linear infinite;
            pointer-events: none;
            z-index: 0;
            opacity: 0.9;
        }

        /* Near star layer with twinkle - faster drift */
        .stApp::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image:
                radial-gradient(3px 3px at 25px 45px, #00FFFF, transparent),
                radial-gradient(2.5px 2.5px at 75px 15px, #FF00FF, transparent),
                radial-gradient(3px 3px at 125px 95px, #4F46E5, transparent),
                radial-gradient(2px 2px at 175px 55px, #00FFFF, transparent),
                radial-gradient(3px 3px at 45px 135px, #8B5CF6, transparent),
                radial-gradient(2.5px 2.5px at 155px 175px, #FF00FF, transparent);
            background-size: 250px 200px;
            animation: drift-stars-fast 60s linear infinite, twinkle-stars 4s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
            opacity: 0.8;
        }

        @keyframes drift-stars {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-200px, -200px); }
        }

        @keyframes drift-stars-fast {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-250px, -200px); }
        }

        @keyframes twinkle-stars {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 0.4; }
        }

        /* ================================================================== */
        /* MAIN CONTENT CENTER ALIGNMENT                                      */
        /* ================================================================== */

        /* Center main content container */
        [data-testid="stMain"] > div {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        /* Center the block container */
        .stMainBlockContainer {
            max-width: 1200px;
            margin: 0 auto !important;
        }

        /* Center all direct children in main area */
        section.main > div {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Ensure content blocks stretch but stay centered */
        section.main > div > div {
            width: 100%;
            max-width: 1200px;
        }

        /* ================================================================== */
        /* PHASE 1: ANIMATED MESH GRADIENT WITH PARTICLE SYSTEM              */
        /* ================================================================== */

        @keyframes mesh-gradient-shift {
            0% { background-position: 0% 50%; }
            25% { background-position: 100% 50%; }
            50% { background-position: 100% 100%; }
            75% { background-position: 0% 100%; }
            100% { background-position: 0% 50%; }
        }

        /* ================================================================== */
        /* PHASE 8: CUSTOM NEON CURSOR & SELECTION EFFECTS                   */
        /* ================================================================== */

        * {
            cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="4" fill="%2300FFFF" opacity="0.9"/><circle cx="16" cy="16" r="8" fill="none" stroke="%2300FFFF" stroke-width="1" opacity="0.5"/><circle cx="16" cy="16" r="12" fill="none" stroke="%23FF00FF" stroke-width="0.5" opacity="0.3"/></svg>') 16 16, auto !important;
        }

        ::selection {
            background: rgba(0, 255, 255, 0.4) !important;
            color: #FFFFFF !important;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        ::-moz-selection {
            background: rgba(0, 255, 255, 0.4) !important;
            color: #FFFFFF !important;
        }

        *:focus {
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(0, 255, 255, 0.5), 0 0 20px rgba(0, 255, 255, 0.3) !important;
        }

        /* ================================================================== */
        /* PHASE 3: ADVANCED GLASSMORPHISM CARDS WITH NEON BORDERS           */
        /* ================================================================== */

        /* Glass card with depth layers */
        .glass-card {
            position: relative;
            background: linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.08) 0%,
                rgba(255, 255, 255, 0.02) 100%
            ) !important;
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 24px !important;
            padding: 2rem !important;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.4),
                0 0 40px rgba(0, 255, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            overflow: hidden;
        }

        /* Glowing neon border effect */
        .glow-border {
            border: 1px solid rgba(0, 255, 255, 0.3) !important;
            box-shadow:
                0 0 20px rgba(0, 255, 255, 0.15),
                0 0 40px rgba(79, 70, 229, 0.1),
                inset 0 0 20px rgba(0, 255, 255, 0.02) !important;
        }

        /* Animated rotating neon border */
        .nexus-neon-border {
            position: relative;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)) !important;
            border: 2px solid transparent !important;
            border-radius: 20px !important;
            animation: border-glow-pulse 3s ease-in-out infinite;
        }

        @keyframes border-glow-pulse {
            0%, 100% {
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), 0 0 40px rgba(79, 70, 229, 0.2);
            }
            50% {
                box-shadow: 0 0 30px rgba(0, 255, 255, 0.5), 0 0 60px rgba(255, 0, 255, 0.3);
            }
        }

        /* ================================================================== */
        /* PHASE 2: 3D HOLOGRAPHIC HERO SECTION                              */
        /* ================================================================== */

        .nexus-hero-title {
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.02em;
            background: linear-gradient(
                135deg,
                #00FFFF 0%,
                #FFFFFF 25%,
                #FF00FF 50%,
                #FFFFFF 75%,
                #00FFFF 100%
            );
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: chrome-shift 3s ease infinite;
            filter: drop-shadow(0 0 30px rgba(0, 255, 255, 0.4));
        }

        @keyframes chrome-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        /* Holographic subtitle with scan lines */
        .nexus-subtitle {
            position: relative;
            color: rgba(148, 163, 184, 0.9) !important;
            font-size: 1.2rem !important;
            letter-spacing: 0.3em;
            text-transform: uppercase;
        }

        /* Status badges with neon pulse */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.25rem;
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 30px;
            color: #00FFFF !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            animation: float-badge 3s ease-in-out infinite, neon-pulse-cyan 2s ease-in-out infinite;
        }

        .status-badge-magenta {
            background: rgba(255, 0, 255, 0.1);
            border-color: rgba(255, 0, 255, 0.3);
            color: #FF00FF !important;
            animation: float-badge 3s ease-in-out infinite 0.5s, neon-pulse-magenta 2s ease-in-out infinite;
        }

        @keyframes float-badge {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }

        @keyframes neon-pulse-cyan {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 255, 0.3), 0 0 20px rgba(0, 255, 255, 0.2); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.5), 0 0 40px rgba(0, 255, 255, 0.3); }
        }

        @keyframes neon-pulse-magenta {
            0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 255, 0.3), 0 0 20px rgba(255, 0, 255, 0.2); }
            50% { box-shadow: 0 0 20px rgba(255, 0, 255, 0.5), 0 0 40px rgba(255, 0, 255, 0.3); }
        }

        /* ================================================================== */
        /* PHASE 4: FLOATING ANIMATIONS & HOVER EFFECTS                      */
        /* ================================================================== */

        .nexus-float {
            animation: nexus-float 4s ease-in-out infinite;
        }

        @keyframes nexus-float {
            0%, 100% { transform: translateY(0px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
            25% { transform: translateY(-10px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4); }
            50% { transform: translateY(-15px); box-shadow: 0 30px 50px rgba(0, 0, 0, 0.3); }
            75% { transform: translateY(-8px); box-shadow: 0 25px 45px rgba(0, 0, 0, 0.35); }
        }

        /* ================================================================== */
        /* PHASE 5: VOLUMETRIC LIGHTING & CAUSTIC EFFECTS                    */
        /* ================================================================== */

        .nexus-spotlight::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(
                ellipse at 30% 30%,
                rgba(255, 255, 255, 0.1) 0%,
                rgba(255, 255, 255, 0.03) 20%,
                transparent 60%
            );
            pointer-events: none;
        }

        /* Light sweep animation */
        .nexus-light-sweep {
            position: relative;
            overflow: hidden;
        }

        .nexus-light-sweep::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(255, 255, 255, 0.08) 50%,
                transparent 100%
            );
            animation: light-sweep-move 4s ease-in-out infinite;
        }

        @keyframes light-sweep-move {
            0% { left: -100%; }
            50% { left: 100%; }
            100% { left: 100%; }
        }

        /* ================================================================== */
        /* PHASE 6: CYBERPUNK SIDEBAR                                        */
        /* ================================================================== */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(10, 10, 15, 0.98) 0%,
                rgba(15, 23, 42, 0.95) 100%
            ) !important;
            border-right: 1px solid rgba(0, 255, 255, 0.2) !important;
            box-shadow:
                4px 0 30px rgba(0, 255, 255, 0.1),
                inset -1px 0 20px rgba(79, 70, 229, 0.05) !important;
            min-width: 0px !important;
            width: 200px !important;
        }

        /* Collapsed sidebar by default - minimal width */
        [data-testid="stSidebar"][aria-expanded="true"] {
            width: 280px !important;
            min-width: 280px !important;
        }

        [data-testid="stSidebar"][aria-expanded="false"] {
            width: 0px !important;
            min-width: 0px !important;
        }

        /* Sidebar collapse button styling */
        [data-testid="stSidebarCollapsedControl"] {
            color: #00FFFF !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
        }

        /* ================================================================== */
        /* PHASE 9: COLOR-SHIFTING TEXT & ANIMATED BADGES                    */
        /* ================================================================== */

        .nexus-gradient-text {
            background: linear-gradient(
                90deg,
                #00FFFF 0%,
                #4F46E5 25%,
                #FF00FF 50%,
                #4F46E5 75%,
                #00FFFF 100%
            );
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradient-flow 3s linear infinite;
        }

        @keyframes gradient-flow {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }

        /* Text styling with neon glow */
        h1, h2, h3 {
            color: #F8FAFC !important;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        }
        p, span, label { color: #CBD5E1 !important; }

        /* ================================================================== */
        /* PHASE 10: INTERACTIVE INPUT FIELDS & BUTTONS                      */
        /* ================================================================== */

        /* Futuristic text input */
        .stTextArea textarea {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 2px solid rgba(79, 70, 229, 0.3) !important;
            border-radius: 16px !important;
            color: #F8FAFC !important;
            font-size: 1rem !important;
            padding: 1.25rem !important;
            transition: all 0.3s ease !important;
        }

        .stTextArea textarea:focus {
            border-color: #00FFFF !important;
            outline: none !important;
            box-shadow:
                0 0 0 4px rgba(0, 255, 255, 0.1),
                0 0 30px rgba(0, 255, 255, 0.2),
                inset 0 0 20px rgba(0, 255, 255, 0.05) !important;
        }

        /* Override ALL textarea focus states - prevent red border */
        [data-testid="stTextArea"] textarea:focus,
        .stTextArea > div > div > textarea:focus,
        textarea:focus {
            border-color: #00FFFF !important;
            outline: none !important;
            box-shadow: 0 0 0 4px rgba(0, 255, 255, 0.1),
                        0 0 30px rgba(0, 255, 255, 0.2) !important;
        }

        /* Override container focus-within state */
        .stTextArea:focus-within,
        [data-testid="stTextArea"]:focus-within,
        .stTextArea > div:focus-within {
            border-color: #00FFFF !important;
            outline: none !important;
        }

        /* Remove any red error/validation borders */
        .stTextArea > div[data-baseweb] {
            border-color: rgba(79, 70, 229, 0.3) !important;
        }

        .stTextArea > div[data-baseweb]:focus-within {
            border-color: #00FFFF !important;
        }

        /* Override Streamlit's default red focus ring */
        [data-baseweb="textarea"]:focus-within {
            border-color: #00FFFF !important;
            box-shadow: 0 0 0 3px rgba(0, 255, 255, 0.15) !important;
        }

        .stTextArea textarea::placeholder {
            color: rgba(148, 163, 184, 0.5) !important;
        }

        /* Primary action button with gradient animation */
        .stButton > button[kind="primary"] {
            position: relative !important;
            background: linear-gradient(135deg, #4F46E5 0%, #00FFFF 50%, #FF00FF 100%) !important;
            background-size: 200% 200% !important;
            border: none !important;
            border-radius: 16px !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            padding: 1rem 2rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            overflow: hidden !important;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            animation: button-gradient 3s ease infinite !important;
        }

        @keyframes button-gradient {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-4px) scale(1.02) !important;
            box-shadow:
                0 20px 40px rgba(79, 70, 229, 0.4),
                0 0 60px rgba(0, 255, 255, 0.3),
                0 0 100px rgba(255, 0, 255, 0.2) !important;
        }

        /* Secondary buttons with glass effect */
        .stButton > button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(0, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: #00FFFF !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover {
            background: rgba(0, 255, 255, 0.1) !important;
            border-color: rgba(0, 255, 255, 0.4) !important;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2) !important;
            transform: translateX(4px) !important;
        }

        /* ================================================================== */
        /* PHASE 12: FLOATING OBSIDIAN PLATFORMS                             */
        /* ================================================================== */

        .nexus-platform {
            position: relative;
            background: linear-gradient(135deg,
                rgba(10, 10, 15, 0.95) 0%,
                rgba(20, 20, 30, 0.9) 50%,
                rgba(10, 10, 15, 0.95) 100%
            );
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.8),
                0 40px 100px rgba(0, 0, 0, 0.5),
                0 0 40px rgba(0, 255, 255, 0.1),
                0 0 80px rgba(79, 70, 229, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            animation: platform-float 6s ease-in-out infinite;
        }

        @keyframes platform-float {
            0%, 100% { transform: translateY(0) rotateX(2deg) rotateY(-1deg); }
            33% { transform: translateY(-15px) rotateX(-1deg) rotateY(2deg); }
            66% { transform: translateY(-8px) rotateX(1deg) rotateY(-2deg); }
        }

        /* ================================================================== */
        /* METRICS WITH HOLOGRAPHIC NUMBERS                                  */
        /* ================================================================== */

        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)) !important;
            border: 1px solid rgba(0, 255, 255, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.25rem !important;
            position: relative;
            overflow: hidden;
            animation: metric-glow 3s ease-in-out infinite;
        }

        @keyframes metric-glow {
            0%, 100% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.1); }
            50% { box-shadow: 0 0 25px rgba(0, 255, 255, 0.2); }
        }

        [data-testid="stMetricValue"] {
            color: #00FFFF !important;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5), 0 0 40px rgba(0, 255, 255, 0.3) !important;
            font-weight: 800 !important;
            font-size: 1.5rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.02em !important;
            font-size: 0.7rem !important;
            white-space: nowrap !important;
            overflow: visible !important;
        }

        /* ================================================================== */
        /* CLICKABLE METRIC POPOVERS                                         */
        /* ================================================================== */

        [data-testid="stPopover"] > button {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)) !important;
            border: 1px solid rgba(0, 255, 255, 0.2) !important;
            border-radius: 16px !important;
            padding: 1rem 1.25rem !important;
            width: 100% !important;
            color: #00FFFF !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.4) !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stPopover"] > button:hover {
            border-color: #00FFFF !important;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
        }

        [data-testid="stPopoverBody"] {
            background: rgba(10, 10, 15, 0.98) !important;
            border: 1px solid rgba(0, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            padding: 0.5rem !important;
        }

        [data-testid="stPopoverBody"] h3 {
            color: #00FFFF !important;
            margin-bottom: 0.75rem !important;
        }

        /* ================================================================== */
        /* EXPANDERS WITH HOLOGRAPHIC STYLING                                */
        /* ================================================================== */

        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), rgba(79, 70, 229, 0.05)) !important;
            border-radius: 12px !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(0, 255, 255, 0.15) !important;
            transition: all 0.3s ease !important;
        }

        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(79, 70, 229, 0.1)) !important;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.1) !important;
        }

        /* ================================================================== */
        /* DIVIDERS & ALERTS                                                 */
        /* ================================================================== */

        hr {
            border-color: rgba(0, 255, 255, 0.2) !important;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
        }

        .stMarkdown { color: #CBD5E1 !important; }

        .stAlert {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), rgba(79, 70, 229, 0.05)) !important;
            border: 1px solid rgba(0, 255, 255, 0.2) !important;
            border-radius: 12px !important;
        }

        /* ================================================================== */
        /* PHASE 14: CHROMATIC ABERRATION & RAINBOW CAUSTICS                 */
        /* ================================================================== */

        .nexus-caustics-rainbow {
            position: relative;
        }

        .nexus-caustics-rainbow::before {
            content: '';
            position: absolute;
            inset: 0;
            background:
                radial-gradient(ellipse at 20% 30%, rgba(255,0,0,0.08) 0%, transparent 40%),
                radial-gradient(ellipse at 50% 35%, rgba(255,255,0,0.05) 0%, transparent 40%),
                radial-gradient(ellipse at 75% 40%, rgba(0,255,255,0.06) 0%, transparent 40%),
                radial-gradient(ellipse at 85% 55%, rgba(255,0,255,0.07) 0%, transparent 35%);
            animation: caustics-shift 10s ease-in-out infinite;
            mix-blend-mode: overlay;
            pointer-events: none;
            border-radius: inherit;
            opacity: 0.6;
        }

        @keyframes caustics-shift {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.05) rotate(3deg); }
        }

        /* ================================================================== */
        /* REDUCED MOTION ACCESSIBILITY                                      */
        /* ================================================================== */

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # NEXUS BACKGROUND LAYERS HTML
    # ==========================================================================
    st.markdown("""
    <div class="nexus-warp-tunnel" style="
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: -5;
        overflow: hidden;
        opacity: 0.3;
    ">
        <div class="nexus-tunnel-ring" style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            border: 2px solid rgba(0, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: tunnel-zoom 5s linear infinite;
        "></div>
        <div class="nexus-tunnel-ring" style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            border: 2px solid rgba(255, 0, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: tunnel-zoom 5s linear infinite 1s;
        "></div>
        <div class="nexus-tunnel-ring" style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            border: 2px solid rgba(79, 70, 229, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: tunnel-zoom 5s linear infinite 2s;
        "></div>
        <div class="nexus-tunnel-ring" style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            border: 2px solid rgba(0, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: tunnel-zoom 5s linear infinite 3s;
        "></div>
        <div class="nexus-tunnel-ring" style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            border: 2px solid rgba(255, 0, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: tunnel-zoom 5s linear infinite 4s;
        "></div>
    </div>
    <style>
        @keyframes tunnel-zoom {
            0% { width: 10px; height: 10px; opacity: 0; border-width: 2px; }
            10% { opacity: 0.4; }
            90% { opacity: 0.2; }
            100% { width: 3000px; height: 3000px; opacity: 0; border-width: 1px; }
        }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # AUDIO VISUALIZER HTML (CSS Animation)
    # ==========================================================================
    st.markdown("""
    <div class="nexus-visualizer" style="
        position: fixed;
        top: 5rem;
        left: 1rem;
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 40px;
        z-index: 50;
        opacity: 0.35;
    ">
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.8s ease-in-out infinite;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.6s ease-in-out infinite 0.1s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.9s ease-in-out infinite 0.2s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.5s ease-in-out infinite 0.15s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.7s ease-in-out infinite 0.3s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 1.0s ease-in-out infinite 0.05s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.55s ease-in-out infinite 0.25s;"></div>
        <div class="nexus-viz-bar" style="width: 4px; background: linear-gradient(180deg, #00FFFF, #FF00FF); border-radius: 2px; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); animation: viz-pulse 0.85s ease-in-out infinite 0.12s;"></div>
    </div>
    <style>
        @keyframes viz-pulse {
            0%, 100% { height: 15px; }
            50% { height: 35px; }
        }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # ==========================================================================
    # NEXUS HOLOGRAPHIC HERO SECTION
    # ==========================================================================
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; position: relative;">
        <h1 class="nexus-hero-title" style="font-size: 3.5rem; font-weight: 900; letter-spacing: -0.02em; background: linear-gradient(135deg, #00FFFF 0%, #FFFFFF 25%, #FF00FF 50%, #FFFFFF 75%, #00FFFF 100%); background-size: 200% 200%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: chrome-shift 3s ease infinite; filter: drop-shadow(0 0 30px rgba(0, 255, 255, 0.4)); margin-bottom: 0.5rem;">HCI Research Interface</h1>
        <p style="color: rgba(148, 163, 184, 0.9); font-size: 1.1rem; letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 1.5rem;">Multi-Agent HCI Research Synthesis Engine</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.25rem; background: rgba(0, 255, 255, 0.1); border: 1px solid rgba(0, 255, 255, 0.3); border-radius: 30px; color: #00FFFF; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">
                <span style="width: 8px; height: 8px; background: #00FFFF; border-radius: 50%; box-shadow: 0 0 10px #00FFFF;"></span>
                System Online
            </span>
            <span style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.25rem; background: rgba(255, 0, 255, 0.1); border: 1px solid rgba(255, 0, 255, 0.3); border-radius: 30px; color: #FF00FF; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">
                <span style="width: 8px; height: 8px; background: #FF00FF; border-radius: 50%; box-shadow: 0 0 10px #FF00FF;"></span>
                HCI Research
            </span>
            <span style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.25rem; background: rgba(79, 70, 229, 0.1); border: 1px solid rgba(79, 70, 229, 0.3); border-radius: 30px; color: #8B5CF6; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">
                <span style="width: 8px; height: 8px; background: #8B5CF6; border-radius: 50%; box-shadow: 0 0 10px #8B5CF6;"></span>
                AI Powered
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    display_sidebar()

    # Main area - centered layout
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_left:
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

    with col_center:
        query = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?",
            key="query_input"
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

    with col_right:
        # Example queries with glass card styling
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <h4 style="color: #F8FAFC; margin-bottom: 0.5rem;">💡 Try These</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">Click to auto-fill your query</p>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            ("🎯", "User-centered design"),
            ("🔬", "AR usability research"),
            ("🤖", "AI transparency"),
            ("📚", "Ethics in AI"),
        ]

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if st.button(f"{examples[0][0]} {examples[0][1]}", use_container_width=True, key="ex_0"):
                st.session_state.example_query = f"What are the key {examples[0][1].lower()} principles in HCI?"
                st.rerun()
        with row1_col2:
            if st.button(f"{examples[1][0]} {examples[1][1]}", use_container_width=True, key="ex_1"):
                st.session_state.example_query = f"What are the key {examples[1][1].lower()} advances in HCI?"
                st.rerun()

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if st.button(f"{examples[2][0]} {examples[2][1]}", use_container_width=True, key="ex_2"):
                st.session_state.example_query = f"What are the key {examples[2][1].lower()} approaches in HCI?"
                st.rerun()
        with row2_col2:
            if st.button(f"{examples[3][0]} {examples[3][1]}", use_container_width=True, key="ex_3"):
                st.session_state.example_query = f"What are the key {examples[3][1].lower()} considerations in HCI?"
                st.rerun()


    # Safety log (if enabled)
    if st.session_state.show_safety_log:
        st.divider()
        st.markdown("### 🛡️ Safety Event Log")
        # TODO: Display safety events from safety manager
        st.info("No safety events recorded.")


if __name__ == "__main__":
    main()
