"""
Context Manager Module
Extracted from the original generator class with no changes to logic.
Handles continuity between chapters by managing previous context and endings.
"""

import os

from utils.file_utils import read_previous_ending


class ContextManager:

    def __init__(self, generator):
        self.generator = generator

    def get_previous_context(self, chapter_num: int) -> str:
        """Get context from previous chapters for continuity"""
        if chapter_num <= 1:
            return ""

        context_parts = []

        # Get last 2-3 chapter summaries
        start_idx = max(0, len(self.generator.chapter_summaries) - 3)
        recent = self.generator.chapter_summaries[start_idx:]

        if recent:
            context_parts.append("पिछले अध्यायों का सारांश:")
            for summary in recent:
                context_parts.append(f"अध्याय {summary['chapter_num']}: {summary['title']}")
                context_parts.append(f"- {summary['summary'][:300]}...")
                context_parts.append(f"- अंत: {summary['ending']}")
                context_parts.append("")

        # Read last chapter's saved ending
        ending = read_previous_ending(
            chapter_num,
            self.generator.CONTEXT_DIR
        )

        if ending:
            context_parts.append("पिछले अध्याय के अंतिम पैराग्राफ:")
            context_parts.append(ending)

        return "\n".join(context_parts)
