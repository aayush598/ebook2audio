"""
Chapter Content Builder Module
Extracted from the original HindiManhwaGenerator class without any changes to logic.
Handles creation of full chapter scripts with previous-context awareness.
"""

import os
import time

from utils.cleaning_utils import deep_clean_for_tts
from utils.file_utils import save_chapter_ending


class ChapterContentBuilder:

    def __init__(self, generator):
        self.generator = generator

    def generate_chapter_content(self, chapter_num: int):
        """Generate full chapter content with context awareness"""

        chapter_outline = next(
            (ch for ch in self.generator.all_chapters if ch.get('chapter_num') == chapter_num),
            None
        )

        if not chapter_outline:
            print(f"❌ Chapter {chapter_num} का outline नहीं मिला")
            return None

        print(f"\n" + "="*60)
        print(f"✍️ अध्याय {chapter_num}: {chapter_outline.get('title', 'Untitled')}")
        print("="*60)

        self.generator._wait_for_rate_limit()

        # Get previous context
        prev_context = self.generator.context_manager.get_previous_context(chapter_num)

        # Build character info
        char_info = "\n".join([
            f"- {c.get('name', 'Unknown')}: {c.get('personality', '')} ({c.get('intelligence_type', 'strategic')})"
            for c in self.generator.series_foundation.get('characters', [])
        ])

        prompt = f"""अध्याय {chapter_num} का पूरा TTS-ready Hindi script लिखो।

सीरीज़: {self.generator.series_foundation['series_title']}
मुख्य कहानी: {self.generator.series_foundation.get('main_storyline', '')}

किरदार (सभी genius level):
{char_info}

{prev_context}

इस अध्याय की जानकारी:
- शीर्षक: {chapter_outline.get('title', '')}
- सीख: {chapter_outline.get('lesson_focus', '')}
- कहानी: {chapter_outline.get('plot_summary', '')}
- दृश्य: {chapter_outline.get('key_scenes', '')}
- स्मार्ट moments: {chapter_outline.get('smart_moments', '')}
- किरदार फोकस: {chapter_outline.get('character_focus', '')}
- अंत: {chapter_outline.get('cliffhanger', '')}

महत्वपूर्ण निर्देश:
1. 6000-8000 शब्दों का विस्तृत script (20-25 मिनट audio)
2. बोलचाल की आधुनिक हिंदी - पुराने शब्द नहीं
3. इंग्लिश नाम/टर्म को देवनागरी में (मार्कस, स्ट्रैटिजी, कमांडर)
4. प्रवाह के लिए अल्पविराम (,) का खूब इस्तेमाल
5. हर दृश्य को विस्तार से बताओ
6. किरदारों की बातचीत बेहद स्मार्ट और clever हो
7. कोई सिंबल नहीं (**, *, ##, (), [])
8. सबक अंत में (5-8 लाइन)
9. पिछले अध्याय से continuity maintain करो
10. कोई जानकारी repeat मत करो, आगे बढ़ाओ

फॉर्मेट:
अध्याय {chapter_num}: {chapter_outline.get('title', '')}

[यहाँ 6000-8000 शब्दों की विस्तृत कहानी]

इस अध्याय से सीख
1. पहली सीख
2. दूसरी सीख
...

अब पूरा अध्याय लिखो।"""

        print("   📝 Content generate हो रहा है...")

        response = self.generator.content_writer.run(
            prompt,
            stream=False,
            user_id=self.generator.session_id
        )
        self.generator.rate_limiter.record_request()

        content = response.content.strip()
        content = deep_clean_for_tts(content)

        # Save chapter ending for next chapter context
        save_chapter_ending(
            chapter_num,
            content,
            self.generator.CONTEXT_DIR
        )

        # Store summary for context
        self.generator.chapter_summaries.append({
            'chapter_num': chapter_num,
            'title': chapter_outline.get('title', ''),
            'summary': chapter_outline.get('plot_summary', ''),
            'ending': chapter_outline.get('cliffhanger', '')
        })

        # Save chapter file
        safe_title = self.generator.series_foundation.get('series_title', 'manhwa').replace(' ', '_')[:30]
        filename = f"{safe_title}_ch{chapter_num:03d}.txt"
        filepath = os.path.join(self.generator.OUTPUT_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        word_count = len(content.split())
        print(f"   ✅ अध्याय {chapter_num} तैयार!")
        print(f"   📊 शब्द: {word_count:,} (~{word_count*0.003:.1f} मिनट)")
        print(f"   💾 Saved: {filename}")

        return content

    def generate_all_chapters(self, start_from: int = 1):
        """Generate all chapters one by one"""

        print("\n" + "="*60)
        print("🚀 सभी अध्याय generate करना शुरू...")
        print("="*60)

        total = len(self.generator.all_chapters)
        success = 0
        failed = []

        for ch in self.generator.all_chapters:
            ch_num = ch.get('chapter_num', 0)

            if ch_num < start_from:
                continue

            print(f"\n[{ch_num}/{total}] Processing...")

            try:
                content = self.generate_chapter_content(ch_num)
                if content:
                    success += 1
                else:
                    failed.append(ch_num)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append(ch_num)
                time.sleep(10)

        print("\n" + "="*60)
        print("📊 Generation Summary")
        print("="*60)
        print(f"   ✅ Successful: {success}/{total}")

        if failed:
            print(f"   ❌ Failed: {failed}")

        print("="*60)

        return success, failed
