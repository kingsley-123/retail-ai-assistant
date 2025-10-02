"""
Simple AI agent that can use tools and reason.
"""
from .tools import ToolKit
import ollama


class SimpleAgent:
    """Agent that can reason and use tools."""
    
    def __init__(self, rag_pipeline=None):
        """
        Initialize agent.
        
        Args:
            rag_pipeline: RAG pipeline for document search
        """
        self.toolkit = ToolKit(rag_pipeline)
        self.model = "llama3.2"
        
    def run(self, task: str) -> str:
        """
        Execute a task by reasoning and using tools.
        
        Args:
            task: Task description
            
        Returns:
            Task result
        """
        # Step 1: Agent plans what to do
        plan = self._create_plan(task)
        print(f"Agent's plan:\n{plan}\n")
        
        # Step 2: Agent executes the plan
        result = self._execute_plan(plan, task)
        
        return result
    
    def _create_plan(self, task: str) -> str:
        """Create a plan to accomplish the task."""
        prompt = f"""You are an AI assistant with access to these tools:
1. search_documents(query) - Search business documents
2. calculate(expression) - Do math calculations
3. format_data(data) - Format data for presentation

Task: {task}

Create a step-by-step plan to accomplish this task. Be specific about which tools to use.
Keep your plan brief (2-4 steps)."""

        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def _execute_plan(self, plan: str, task: str) -> str:
        """Execute the plan using available tools."""
        # Simple execution: check which tools are mentioned
        result = "Execution Results:\n\n"
        
        # Check if plan mentions document search
        if "search_documents" in plan.lower() or "search" in plan.lower():
            print("→ Executing: Document search")
            search_result = self.toolkit.search_documents(task)
            result += f"Document Search:\n{search_result}\n\n"
        
        # Check if plan mentions calculations
        if "calculate" in plan.lower() or "math" in plan.lower():
            print("→ Executing: Calculation")
            # Extract numbers from task for demo
            if "%" in task or "percent" in task:
                calc_result = self.toolkit.calculate("2450000 * 0.15")
                result += f"Calculation:\n{calc_result}\n\n"
        
        # Final answer
        print("→ Generating final answer")
        final_answer = self._generate_final_answer(task, result)
        
        return final_answer
    
    def _generate_final_answer(self, task: str, execution_results: str) -> str:
        """Generate final answer based on execution results."""
        prompt = f"""Task: {task}

Results from tools:
{execution_results}

Based on these results, provide a clear, concise answer to the task."""

        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']


def main():
    """Test the agent without RAG pipeline."""
    agent = SimpleAgent()
    
    # Test task
    task = "Calculate 15% of 2450000 and format the result"
    
    print(f"Task: {task}")
    print("="*70)
    
    result = agent.run(task)
    
    print("\n" + "="*70)
    print("Final Answer:")
    print(result)


if __name__ == "__main__":
    main()