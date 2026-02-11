"""
Full indexing pipeline v3: VTT → Sentences → Topics (with boundary refinement) → Typesense.
Uses topic_segmentation_v3 for commercial-grade segment boundaries.
"""

import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from app.services.sentence_parser import parse_vtt_to_sentences, Sentence
from app.services.topic_segmentation_v3 import segment_sentences, Topic
from app.services.typesense_indexer import index_sentences, index_topics, delete_episode, get_index_stats


@dataclass
class ProcessingResult:
    """Result of processing an episode."""
    episode_id: str
    vtt_path: str
    success: bool
    sentences_count: int = 0
    topics_count: int = 0
    duration_seconds: float = 0
    boundary_refinement: bool = False
    error: Optional[str] = None


def process_episode(
    vtt_path: str, 
    episode_id: Optional[str] = None,
    refine_boundaries: bool = True
) -> ProcessingResult:
    """
    Process a single episode with v3 segmentation (boundary refinement).
    
    Args:
        vtt_path: Path to VTT file
        episode_id: Optional episode ID (defaults to filename)
        refine_boundaries: Whether to run the boundary refinement pass
        
    Returns:
        ProcessingResult with stats
    """
    start_time = time.time()
    vtt_path = Path(vtt_path)
    
    if not episode_id:
        episode_id = vtt_path.stem.replace(".en", "")
    
    result = ProcessingResult(
        episode_id=episode_id,
        vtt_path=str(vtt_path),
        success=False,
        boundary_refinement=refine_boundaries
    )
    
    try:
        # Step 1: Parse VTT to sentences with NER
        print(f"  [1/4] Parsing VTT and extracting entities...")
        sentences = parse_vtt_to_sentences(str(vtt_path))
        result.sentences_count = len(sentences)
        print(f"        Found {len(sentences)} sentences")
        
        if not sentences:
            result.error = "No sentences parsed from VTT"
            return result
        
        # Step 2: Segment into topics with v3 (includes boundary refinement)
        print(f"  [2/4] Segmenting into topics (v3 with boundary refinement={refine_boundaries})...")
        topics = segment_sentences(
            sentences, 
            episode_id, 
            refine_boundaries=refine_boundaries
        )
        result.topics_count = len(topics)
        print(f"        Created {len(topics)} topics")
        
        # Print topic summary
        for i, topic in enumerate(topics):
            duration_min = (topic.end_ms - topic.start_ms) / 60000
            print(f"        [{i}] {duration_min:.1f}min: {topic.summary[:50]}...")
        
        # Step 3: Delete existing data for this episode
        print(f"  [3/4] Clearing existing index data...")
        delete_episode(episode_id)
        
        # Step 4: Index to Typesense
        print(f"  [4/4] Indexing to Typesense...")
        indexed_sentences = index_sentences(sentences, episode_id, topics)
        indexed_topics = index_topics(topics)
        print(f"        Indexed {indexed_sentences} sentences, {indexed_topics} topics")
        
        result.success = True
        result.duration_seconds = time.time() - start_time
        
    except Exception as e:
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
    
    return result


def process_all_episodes(
    audio_dir: str = "audio",
    refine_boundaries: bool = True
) -> List[ProcessingResult]:
    """
    Process all VTT files with v3 segmentation.
    
    Args:
        audio_dir: Directory containing VTT files
        refine_boundaries: Whether to run boundary refinement
        
    Returns:
        List of ProcessingResult for each episode
    """
    audio_path = Path(audio_dir)
    vtt_files = sorted(audio_path.glob("*.vtt"))
    
    print(f"Found {len(vtt_files)} VTT files to process")
    print(f"Boundary refinement: {'ENABLED' if refine_boundaries else 'DISABLED'}")
    print("=" * 60)
    
    results = []
    
    for i, vtt_file in enumerate(vtt_files, 1):
        episode_id = vtt_file.stem.replace(".en", "")
        print(f"\n[{i}/{len(vtt_files)}] Processing: {episode_id}")
        
        result = process_episode(str(vtt_file), episode_id, refine_boundaries)
        results.append(result)
        
        status = "✅" if result.success else "❌"
        print(f"{status} Completed in {result.duration_seconds:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY (v3 Segmentation)")
    print("=" * 60)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Total sentences: {sum(r.sentences_count for r in successful)}")
    print(f"Total topics: {sum(r.topics_count for r in successful)}")
    print(f"Total time: {sum(r.duration_seconds for r in results):.1f}s")
    print(f"Avg time per episode: {sum(r.duration_seconds for r in results)/len(results):.1f}s")
    
    if failed:
        print(f"\nFailed episodes:")
        for r in failed:
            print(f"  - {r.episode_id}: {r.error}")
    
    stats = get_index_stats()
    print(f"\nTypesense index stats: {stats}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        vtt_path = sys.argv[1]
        print(f"Processing single episode: {vtt_path}")
        result = process_episode(vtt_path, refine_boundaries=True)
        print(f"\nResult: {'Success' if result.success else 'Failed'}")
        if result.error:
            print(f"Error: {result.error}")
    else:
        process_all_episodes(refine_boundaries=True)
