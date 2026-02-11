#!/usr/bin/env python3
"""Direct batch re-index without HTTP - more stable."""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.segmentation import index_episode
from app.services import storage

def get_vtt_files():
    """Find all VTT files in audio directory."""
    audio_dir = Path("audio")
    return sorted(audio_dir.glob("*.en.vtt"))

def reindex_episode(vtt_path: Path) -> dict:
    """Re-index a single episode directly."""
    video_id = vtt_path.stem.replace(".en", "")
    vtt_content = vtt_path.read_text()
    episode_title = f"Episode {video_id}"
    
    # Run pipeline
    segments = index_episode(vtt_content, episode_title)
    
    if not segments:
        return {"error": "No segments produced", "video_id": video_id}
    
    # Store
    storage.save_episode(video_id, episode_title, "Unknown Podcast", segments)
    
    return {
        "video_id": video_id,
        "segments": len(segments),
        "durations": [round((s.end_ms - s.start_ms) / 1000, 1) for s in segments]
    }

def main():
    vtt_files = get_vtt_files()
    print(f"Found {len(vtt_files)} VTT files to re-index")
    print("=" * 60)
    
    results = []
    
    for i, vtt_path in enumerate(vtt_files, 1):
        video_id = vtt_path.stem.replace(".en", "")
        print(f"\n[{i}/{len(vtt_files)}] Re-indexing: {video_id}")
        
        start = time.time()
        try:
            result = reindex_episode(vtt_path)
            elapsed = time.time() - start
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
                results.append({"video_id": video_id, "status": "error"})
            else:
                segs = result["segments"]
                durs = result["durations"]
                avg_dur = sum(durs) / len(durs) if durs else 0
                print(f"  ✅ {segs} segments indexed in {elapsed:.1f}s")
                print(f"  📊 Avg duration: {avg_dur:.0f}s, range: {min(durs):.0f}s - {max(durs):.0f}s")
                results.append({"video_id": video_id, "status": "success", "segments": segs})
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({"video_id": video_id, "status": "error", "error": str(e)})
        
        # Small delay
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"✅ Success: {len(success)}")
    print(f"❌ Errors: {len(errors)}")
    
    if success:
        total_segs = sum(r["segments"] for r in success)
        print(f"📊 Total segments: {total_segs}")

if __name__ == "__main__":
    main()
