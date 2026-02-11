#!/usr/bin/env python3
"""Batch re-index all episodes with new 2.5% percentile settings."""

import httpx
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:8765"

def get_vtt_files():
    """Find all VTT files in audio directory."""
    audio_dir = Path("audio")
    return sorted(audio_dir.glob("*.en.vtt"))

def reindex_episode(video_id: str) -> dict:
    """Re-index a single episode."""
    url = f"{BASE_URL}/index/{video_id}"
    
    try:
        # Longer timeout - embedding can be slow on limited hardware
        with httpx.Client(timeout=600.0) as client:
            response = client.post(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text, "status": response.status_code}
    except httpx.TimeoutException as e:
        return {"error": f"Timeout after 600s: {e}", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "exception"}

def main():
    vtt_files = get_vtt_files()
    print(f"Found {len(vtt_files)} VTT files to re-index", flush=True)
    print("=" * 60, flush=True)
    
    results = []
    
    for i, vtt_path in enumerate(vtt_files, 1):
        video_id = vtt_path.stem.replace(".en", "")
        print(f"\n[{i}/{len(vtt_files)}] Re-indexing: {video_id}", flush=True)
        
        start = time.time()
        result = reindex_episode(video_id)
        elapsed = time.time() - start
        
        if "error" in result:
            print(f"  ❌ Error: {result['error'][:100]}")
            results.append({"video_id": video_id, "status": "error", "error": result["error"]})
        else:
            segs = result.get("total_segments", 0)
            print(f"  ✅ {segs} segments indexed in {elapsed:.1f}s")
            
            # Show sample durations
            if result.get("segments"):
                durations = [s["duration_s"] for s in result["segments"]]
                avg_dur = sum(durations) / len(durations)
                print(f"  📊 Avg duration: {avg_dur:.0f}s, range: {min(durations):.0f}s - {max(durations):.0f}s")
            
            results.append({
                "video_id": video_id, 
                "status": "success", 
                "segments": segs,
                "time": elapsed
            })
        
        # Small delay between episodes
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
        total_time = sum(r["time"] for r in success)
        print(f"📊 Total segments: {total_segs}")
        print(f"⏱️  Total time: {total_time:.1f}s")
    
    if errors:
        print("\nFailed episodes:")
        for r in errors:
            print(f"  - {r['video_id']}")

if __name__ == "__main__":
    main()
