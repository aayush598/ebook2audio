"""
Updated main.py — Resume-Capable Version
Automatically resumes from last completed step:
- Foundation
- Chapter Outlines
- Chapter Content

No core generation logic has been touched.
"""

import os
from dotenv import load_dotenv

from config.settings import DEFAULT_MODEL
from generator.hindi_manhwa_generator import HindiManhwaGenerator


def main():
    """Main entry point - resume-capable terminal interaction"""
    print("\n" + "="*60)
    print("📚 Hindi Educational Manhwa Content Generator (Resume Enabled)")
    print("   विस्तृत, संदर्भ-जागरूक हिंदी ऑडियोबुक स्क्रिप्ट्स")
    print("="*60)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env file!")
        print("   Please create a .env file with:")
        print('   GEMINI_API_KEY="your-api-key-here"')
        return
    
    print("\n✅ API Key loaded from .env")

    # For simplicity, topic is fixed (same as original)
    print("\n" + "-"*40)
    topic = "Strategic thinking"
    print(f"🎯 Selected Topic: {topic}")

    print("\n🔧 Generator initialize हो रहा है...")
    generator = HindiManhwaGenerator(
        gemini_api_key=api_key,
        model_id=DEFAULT_MODEL
    )

    # ----------------------------------------------------
    # 1️⃣ Foundation (resume-capable)
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("📌 चरण 1: Foundation Load / Generate")

    foundation = generator.generate_series_foundation(topic)

    if not foundation:
        print("❌ Foundation generation failed!")
        return

    print("\n📖 Foundation Loaded:")
    print(f"   Title: {foundation.get('series_title', 'N/A')}")
    print(f"   Characters: {len(foundation.get('characters', []))}")

    # ----------------------------------------------------
    # 2️⃣ Chapter Outlines (resume-capable)
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("📌 चरण 2: Chapter Outlines Load / Generate")

    outlines = generator.generate_all_chapter_outlines()
    if not outlines:
        print("❌ Chapter outlines generation failed!")
        return
    print(f"\n📚 कुल {len(outlines)} outlines तैयार (resume-supported).")

    # ----------------------------------------------------
    # 3️⃣ Ask User: Generate Chapters or Exit
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("Options:")
    print("   1. सभी chapters generate करें (resume auto)")
    print("   2. Specific range generate करें")
    print("   3. Single chapter generate करें")
    print("   4. Exit")

    choice = input("\nChoice (1-4): ").strip()

    # ----------------------------------------------------
    # 3.1 Generate all chapters with resume
    # ----------------------------------------------------
    if choice == '1':
        success, failed = generator.generate_all_chapters(start_from=1)

    # ----------------------------------------------------
    # 3.2 Generate a specific range
    # ----------------------------------------------------
    elif choice == '2':
        start = int(input("Start chapter: ").strip() or "1")
        end = int(input("End chapter: ").strip() or "10")

        success = 0
        failed = []
        for ch_num in range(start, min(end + 1, len(outlines) + 1)):
            content = generator.generate_chapter_content(ch_num)
            if content:
                success += 1
            else:
                failed.append(ch_num)

        print(f"\n✅ Generated {success} chapters")
        if failed:
            print(f"❌ Failed: {failed}")

    # ----------------------------------------------------
    # 3.3 Generate a single chapter
    # ----------------------------------------------------
    elif choice == '3':
        ch_num = int(input("Chapter number: ").strip() or "1")
        content = generator.generate_chapter_content(ch_num)
        if content:
            print(f"\n✅ Chapter {ch_num} generated successfully!")

    # ----------------------------------------------------
    # 3.4 Exit
    # ----------------------------------------------------
    else:
        print("\n👋 Exiting. All outlines/progress saved automatically.")
        return

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("🎉 Generation Complete! (Resume-enabled)")
    print(f"   📁 Content saved in: {generator.OUTPUT_DIR}/")
    print(f"   📁 Metadata saved in: {generator.METADATA_DIR}/")
    print(f"   📁 Context saved in: {generator.CONTEXT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
