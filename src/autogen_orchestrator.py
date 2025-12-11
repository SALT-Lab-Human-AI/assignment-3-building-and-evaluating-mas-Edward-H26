"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team
from src.agents.memory import ResearchMemory


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Initialize multi-agent memory system
        self.logger.info("Initializing memory system...")
        self.memory = ResearchMemory(max_findings=20, max_context=10)

        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)

        self.logger.info("Research team created successfully")

        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")

        # Get relevant past context from memory
        past_context = self.memory.get_relevant_context(query, k=3)
        if past_context:
            self.logger.info("Found relevant past research in memory")

        try:
            # Run the async query processing with memory context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(query, max_rounds, past_context)
                    ).result()
            else:
                result = loop.run_until_complete(
                    self._process_query_async(query, max_rounds, past_context)
                )

            # Store findings in memory for future context
            self._store_in_memory(query, result)

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(
        self, query: str, max_rounds: int = 20, past_context: str = ""
    ) -> Dict[str, Any]:
        """
        Async implementation of query processing.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            past_context: Relevant context from previous research queries

        Returns:
            Dictionary containing results
        """
        # Get related citations from memory
        related_citations = self.memory.get_related_citations(query, k=5)

        # Create task message with enhanced memory context
        context_section = ""
        if past_context or related_citations:
            context_section = "\n## Memory Context"

            if past_context:
                context_section += f"""
### Relevant Past Research
The following context from previous research may be helpful:

{past_context}
"""

            if related_citations:
                citation_list = "\n".join([
                    f"- {c['title']} ({c.get('year', 'n.d.')})"
                    for c in related_citations
                ])
                context_section += f"""
### Previously Found Relevant Sources
{citation_list}
"""

            context_section += "\nPlease build upon this knowledge where relevant and avoid redundant research.\n"

        task_message = f"""Research Query: {query}
{context_section}
Please work together to answer this query comprehensively:
1. Planner: Create a research plan
2. Researcher: Gather evidence from web and academic sources
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate the quality and provide feedback"""
        
        # Run the team
        result = await self.team.run(task=task_message)
        
        # Extract conversation history
        messages = []
        async for message in result.messages:
            msg_dict = {
                "source": message.source,
                "content": message.content if hasattr(message, 'content') else str(message),
            }
            messages.append(msg_dict)
        
        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break
        
        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")
        
        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
            }
        }

    def _store_in_memory(self, query: str, result: Dict[str, Any]) -> None:
        """
        Store query results in memory for future context.

        Args:
            query: The original research query
            result: The result dictionary from processing
        """
        if "error" in result:
            return  # Don't store failed queries

        response = result.get("response", "")
        sources = self._extract_sources(result)

        # Store the finding in memory
        self.memory.add_finding(query, response, sources)

        # Store citations for deduplication
        for source in sources:
            self.memory.add_citation(source, {"url": source, "query": query})

        # Update context window with key messages
        for msg in result.get("conversation_history", [])[-5:]:
            self.memory.update_context_window(
                f"{msg.get('source', 'Unknown')}: {msg.get('content', '')[:200]}"
            )

        self.logger.info(f"Stored finding in memory. Total findings: {len(self.memory.findings)}")

    def _extract_sources(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract source URLs from conversation history.

        Args:
            result: The result dictionary from processing

        Returns:
            List of source URLs found in the conversation
        """
        sources = []
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            urls = url_pattern.findall(content)
            for url in urls:
                # Clean up URL (remove trailing punctuation)
                url = url.rstrip('.,;:\'")]}')
                if url not in sources:
                    sources.append(url)

        return sources[:20]  # Limit to 20 sources

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.

        Returns:
            Dictionary with memory statistics
        """
        return self.memory.get_statistics()

    def clear_memory(self) -> None:
        """Clear all stored memory."""
        self.memory.clear()
        self.logger.info("Memory cleared")

    def save_memory(self, filepath: str) -> None:
        """
        Save memory state to a file.

        Args:
            filepath: Path to save the memory state
        """
        self.memory.save_to_file(filepath)
        self.logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> bool:
        """
        Load memory state from a file.

        Args:
            filepath: Path to load the memory state from

        Returns:
            True if loaded successfully, False otherwise
        """
        success = self.memory.load_from_file(filepath)
        if success:
            self.logger.info(f"Memory loaded from {filepath}")
        else:
            self.logger.warning(f"Failed to load memory from {filepath}")
        return success

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

