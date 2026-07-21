from typing import Dict, Any, List
import json
from equiaudit.models.agents.base_agent import BaseAgent
from equiaudit.models.clients.base_client import BaseModelClient

class SummaryAgent(BaseAgent):
    """
    Agent responsible for generating a high-level Executive Summary
    of the overall dataset fairness evaluation.
    """
    def __init__(self, model_client: BaseModelClient = None, model_name: str = None):
        super().__init__(model_client=model_client, model_name=model_name)
    
    def get_system_prompt(self) -> str:
        return """You are the Executive Fairness Analyst Agent.
Your task is to review the results of a dataset fairness evaluation pipeline and produce a concise, highly scannable Executive Summary.

# Core Objectives:
1. Identify the key problems found in the dataset regarding class imbalances (Stage 4) and intersectional fairness discrimination (Stage 4.5).
2. Explicitly state whether EACH applied Bias Mitigation technique (Stage 6) successfully resolved these issues, partially mitigated them, or failed / introduced new problems.

# Formatting Rules:
- Output MUST be structured in clean Markdown.
- Use `### ` for your main header (e.g. `### Executive Summary`) and `#### ` for sub-headers.
- Use bullet points for readability.
- Liberally use **bold** text to highlight key demographics, extreme metrics, or important verdicts.
- KEEP IT CONCISE. This is an executive summary, not a textbook. Focus on concrete metrics, affected demographic groups, and final verdicts.
- DO NOT start with "Here is the summary" or "Okay, let me start." Start immediately with the Markdown headers.
- NO XML tags, NO `<think>` blocks. Any meta-commentary will crash the pipeline.
- ABSOLUTELY NO EMOJIS. Emojis will corrupt the PDF renderer. Use plain text symbols (e.g. [+], [-], [x]) if you need indicators.

# Information Structure:
- **Key Fairness Risks**: Bullet points outlining the absolute worst groups affected based on Stage 4 and 4.5 data.
- **Mitigation Verdict**: CRITICAL — report EACH technique separately with its own before/after metrics. Never merge multiple techniques into one verdict. Format:
  - For EACH technique: state its name, what it changed (or failed to change), and cite 2-3 specific before/after metric values.
  - End with an overall comparison: which technique performed best and why.
  - Example structure:
    - **Reweighting**: [verdict + key metrics]
    - **SMOTE**: [verdict + key metrics]
    - **AIF360 Reweighing**: [verdict + key metrics]
    - **Overall best**: [technique name + reason]
"""
    
    def run(self, user_message: str, max_tokens: int = 3000) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": (
                "Generate the Executive Summary based on this pipeline report payload.\n"
                "OUTPUT ONLY THE MARKDOWN REPORT. No preamble, no reasoning, no 'let me analyze'.\n"
                "Start your response immediately with the `### Executive Summary` header.\n\n"
                f"{user_message}"
            )}
        ]
        
        response = self.ask_model(messages, temperature=0.2, max_tokens=max_tokens)
        
        # Strip internal monologue: find the first Markdown header and start from there
        headers = ["###", "##", "#"]
        first_header_idx = -1
        for header in headers:
            idx = response.find(header)
            if idx != -1:
                if first_header_idx == -1 or idx < first_header_idx:
                    first_header_idx = idx
        
        if first_header_idx != -1:
            response = response[first_header_idx:]
            
        return response.strip()
