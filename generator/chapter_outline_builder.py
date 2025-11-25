"""
Chapter Outline Builder Module
Extracted from the original HindiManhwaGenerator class with NO changes to logic.
Handles outline generation for all 100 chapters in batches.
"""

import os
import json
import time

from utils.json_utils import extract_json


class ChapterOutlineBuilder:

    def __init__(self, generator):
        self.generator = generator

    def generate_chapter_batch(self, start_ch: int, end_ch: int):
        """Generate chapter outlines for a batch"""
        print(f"\n📚 Chapters {start_ch}-{end_ch} का outline बना रहे हैं...")

        self.generator._wait_for_rate_limit()

        difficulty = (
            "शुरुआती" if start_ch <= 20 else
            "मध्यम" if start_ch <= 50 else
            "उन्नत" if start_ch <= 75 else
            "विशेषज्ञ"
        )

        char_names = ', '.join([
            f"{c.get('name', 'Unknown')} ({c.get('role', 'N/A')})"
            for c in self.generator.series_foundation.get('characters', [])[:5]
        ])

        prompt = f"""सीरीज़ "{self.generator.series_foundation['series_title']}" के अध्याय {start_ch} से {end_ch} का outline बनाओ।

सीरीज़ संदर्भ:
- मुख्य कहानी: {self.generator.series_foundation.get('main_storyline', '')}
- केंद्रीय संघर्ष: {self.generator.series_foundation.get('central_conflict', '')}
- किरदार: {char_names}

JSON array return करो:
[
    {{
        "chapter_num": {start_ch},
        "title": "अध्याय का शीर्षक (देवनागरी में)",
        "lesson_focus": "इस अध्याय में मुख्य सीख (2-3 वाक्य)",
        "plot_summary": "मुख्य घटनाएं (5-6 वाक्य, विस्तार से)",
        "character_focus": "किस किरदार का विकास होगा",
        "key_scenes": "4-5 महत्वपूर्ण दृश्य",
        "smart_moments": "किरदारों के बुद्धिमत्ता वाले पल",
        "cliffhanger": "अगले अध्याय के लिए suspense",
        "difficulty": "{difficulty}"
    }}
]

{end_ch - start_ch + 1} अध्यायों का outline बनाओ।
सिर्फ JSON array, कोई markdown नहीं।"""

        response = self.generator.story_planner.run(
            prompt,
            stream=False,
            user_id=self.generator.session_id
        )
        self.generator.rate_limiter.record_request()

        clean = extract_json(response.content.strip())

        try:
            chapters = json.loads(clean)
            if isinstance(chapters, dict):
                chapters = [chapters]

            valid = [
                ch for ch in chapters
                if isinstance(ch, dict) and 'chapter_num' in ch
            ]

            print(f"   ✅ {len(valid)} chapters का outline तैयार")
            return valid

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON Error: {e}")
            return []

    def generate_all_chapter_outlines(self):
        """Generate all 100 chapter outlines in batches"""
        print("\n" + "="*60)
        print("📚 सभी 100 अध्यायों का outline बना रहे हैं...")
        print("="*60)

        batches = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
        all_chapters = []

        for idx, (start, end) in enumerate(batches):
            print(f"\n🔄 Batch {idx+1}/5: Chapters {start}-{end}")
            batch = self.generate_chapter_batch(start, end)

            if batch:
                all_chapters.extend(batch)
            else:
                print(f"   ⚠️ Batch {idx+1} failed, retrying...")
                time.sleep(5)
                batch = self.generate_chapter_batch(start, end)
                if batch:
                    all_chapters.extend(batch)

        self.generator.all_chapters = all_chapters

        # Save all outlines
        filepath = os.path.join(
            self.generator.METADATA_DIR,
            f"{self.generator.session_id}_all_chapters.json"
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'foundation': self.generator.series_foundation,
                'chapters': all_chapters,
                'total': len(all_chapters)
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✅ कुल {len(all_chapters)} अध्यायों का outline तैयार!")
        print(f"💾 Saved: {filepath}")

        return all_chapters
