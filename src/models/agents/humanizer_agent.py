from models.agents.base_agent import BaseAgent
from models.clients.base_client import BaseModelClient

import os

class HumanizerAgent(BaseAgent):
    def __init__(self, model_client: BaseModelClient = None, model_name: str = None):
        super().__init__(model_client=model_client, model_name=model_name)
    
    def get_system_prompt(self) -> str:
        skill_path = os.path.join(os.path.dirname(__file__), "SKILL.md")
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                skill_content = f.read()
        except Exception:
            skill_content = "You are a writing editor that identifies and removes signs of AI-generated text."

        # Strip everything after "## Process" so we don't adopt the drafting/scratchpad behavior
        if "## Process" in skill_content:
            skill_content = skill_content.split("## Process")[0]

        return f"""{skill_content}

---
YOUR SPECIFIC TASK FOR THIS PIPELINE:
Your primary task is to receive an AI-generated technical report module and rewrite it.

CRITICAL OVERRIDE OF ABOVE WIKIPEDIA RULES:
- You are writing a TECHNICAL REPORT, not a Wikipedia article or an essay.
- SCANNABILITY IS YOUR TOP PRIORITY.
- Therefore, you MUST KEEP and ENHANCE bullet points, numbered lists, section headers, and **bold text** for keywords and key metrics. Do not ever destroy structural bullet points or flatten them into prose.
- You must structure the text with rich Markdown, extensively using **bolding** and *italics* to emphasize key insights, metrics, and organizational hierarchy.
- **Header Formatting**: Organize content cleanly into a clear hierarchy. Use `## ` for main section titles and `### ` or `#### ` for subtitles. Do NOT use `# ` (Level 1) headers since they conflict with the report's global title.
- Your only goal from the rules above is to strip out flowery AI "soul" (like "delve", "pivotal", "stands as a testament"), repetitive filler, vague attributions, and empty positive conclusions.

Additionally, if the generated text repeats itself, states the same point multiple times, or is overly verbose, condense it so that it is concise, direct, and non-repetitive while maintaining the core facts.

EXTREME AND UNBREAKABLE RUELS:
1. OUTPUT ONLY THE FINAL REWRITTEN TEXT AND ABSOLUTELY NOTHING ELSE. 
2. DO NOT START YOUR OUTPUT WITH ANY INTRODUCTORY OR CHATTY PHRASES (e.g., "Here is the humanized text:", "Okay, let me start...", "First, I'll scan...").
3. DO NOT OUTPUT YOUR INTERNAL THOUGHT PROCESS. Start immediately with the first word of the humanized report. Any meta-commentary will instantly crash the automated software pipeline viewing your text.
"""
    
    def run(self, user_message: str, max_tokens: int = 4096) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": (
                "OUTPUT ONLY THE FINAL REWRITTEN TEXT. "
                "Begin your response with the very first word or Markdown element of the rewritten content. "
                "Any preamble, internal reasoning, or meta-commentary will corrupt the automated pipeline and must NEVER appear.\n\n"
                f"Text to humanize:\n\n{user_message}"
            )}
        ]
        
        response = self.ask_model(messages, temperature=0.4, max_tokens=max_tokens)
        # Apply a secondary inline reasoning strip for any leakage that got past ask_model
        response = self._strip_inline_reasoning(response)
        return response
