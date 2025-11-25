"""
Entry point for the Hindi Educational Manhwa Content Generator - Terminal Version
This file wires up the refactored modules and runs the original main() logic unchanged.
"""

import os
from dotenv import load_dotenv

from config.settings import DEFAULT_MODEL
from generator.hindi_manhwa_generator import HindiManhwaGenerator

def main():
    """Main entry point - terminal interaction"""
    print("\n" + "="*60)
    print("📚 Hindi Educational Manhwa Content Generator")
    print("   विस्तृत, संदर्भ-जागरूक हिंदी ऑडियोबुक स्क्रिप्ट्स")
    print("="*60)
    
    # Get API key from environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env file!")
        print("   Please create a .env file with:")
        print('   GEMINI_API_KEY="your-api-key-here"')
        return
    
    print(f"\n✅ API Key loaded from .env")
    
    # Get topic from user
    print("\n" + "-"*40)
    # topic = input("📝 कहानी का विषय दर्ज करें (Topic): ").strip()
    topic = "Strategic thinking"
    
    if not topic:
        print("❌ कोई विषय नहीं दिया गया!")
        return
    
    print(f"\n🎯 Selected Topic: {topic}")
    
    # Confirm
    # confirm = input("\n▶️ Generation शुरू करें? (y/n): ").strip().lower()
    # if confirm != 'y':
    #     print("❌ Cancelled by user")
    #     return
    
    # Initialize generator
    print("\n🔧 Generator initialize हो रहा है...")
    generator = HindiManhwaGenerator(
        gemini_api_key=api_key,
        model_id=DEFAULT_MODEL
    )
    
    # Step 1: Generate foundation
    foundation = generator.generate_series_foundation(topic)
    if not foundation:
        print("❌ Foundation generation failed!")
        return
    
    print(f"Foundation : {foundation}")
    
    # Display foundation info
    print("\n" + "-"*40)
    print("📖 Series Details:")
    print(f"   Title: {foundation.get('series_title', 'N/A')}")
    print(f"   Topic: {foundation.get('skill_topic', topic)}")
    print(f"\n   Story Overview:")
    overview = foundation.get('story_overview', 'N/A')[:500]
    print(f"   {overview}...")
    
    print(f"\n   👥 Characters ({len(foundation.get('characters', []))}):")
    for char in foundation.get('characters', [])[:5]:
        print(f"      - {char.get('name', 'Unknown')}: {char.get('role', 'N/A')}")
        print(f"        Intelligence: {char.get('intelligence_type', 'strategic')}")
    
    # Confirm to continue
    # cont = input("\n▶️ Chapter outlines generate करें? (y/n): ").strip().lower()
    # if cont != 'y':
    #     print("⏸️ Stopped. Foundation saved.")
    #     return
    
    # Step 2: Generate all chapter outlines
    chapters = generator.generate_all_chapter_outlines()
    if not chapters:
        print("❌ Chapter outlines generation failed!")
        return
    
    print(f"\n📚 {len(chapters)} अध्यायों का outline तैयार")
    
    # Ask how many chapters to generate
    print("\n" + "-"*40)
    print("Options:")
    print("   1. सभी chapters generate करें (1-100)")
    print("   2. Specific range generate करें")
    print("   3. Single chapter generate करें")
    print("   4. Exit (outlines saved)")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        success, failed = generator.generate_all_chapters(start_from=1)
    elif choice == '2':
        start = int(input("Start chapter: ").strip() or "1")
        end = int(input("End chapter: ").strip() or "10")
        
        success = 0
        failed = []
        for ch_num in range(start, min(end + 1, len(chapters) + 1)):
            content = generator.generate_chapter_content(ch_num)
            if content:
                success += 1
            else:
                failed.append(ch_num)
        
        print(f"\n✅ Generated {success} chapters")
        if failed:
            print(f"❌ Failed: {failed}")
    elif choice == '3':
        ch_num = int(input("Chapter number: ").strip() or "1")
        content = generator.generate_chapter_content(ch_num)
        if content:
            print(f"\n✅ Chapter {ch_num} generated successfully!")
    else:
        print("👋 Exiting. All outlines have been saved.")
        return
    
    print("\n" + "="*60)
    print("🎉 Generation Complete!")
    print(f"   📁 Content saved in: {generator.OUTPUT_DIR}/")
    print(f"   📁 Metadata saved in: {generator.METADATA_DIR}/")
    print(f"   📁 Context saved in: {generator.CONTEXT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
