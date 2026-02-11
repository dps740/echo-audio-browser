"""
Full indexing pipeline: VTT → Sentences → Topics → Typesense.
Orchestrates all components for processing episodes.
"""

import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from app.services.sentence_parser import parse_vtt_to_sentences, Sentence
from app.services.topic_segmentation_v2 import segment_sentences, Topic
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
    error: Optional[str] = None


def process_episode(vtt_path: str, episode_id: Optional[str] = None) -> ProcessingResult:
    """
    Process a single episode: parse VTT, extract entities, segment topics, index to Typesense.
    
    Args:
        vtt_path: Path to VTT file
        episode_id: Optional episode ID (defaults to filename without extension)
        
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
        success=False
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
        
        # Step 2: Segment into topics
        print(f"  [2/4] Segmenting into topics...")
        topics = segment_sentences(sentences, episode_id)
        result.topics_count = len(topics)
        print(f"        Created {len(topics)} topics")
        
        # Step 3: Delete existing data for this episode (if re-indexing)
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
        print(f"  ERROR: {e}")
    
    return result


def process_all_episodes(audio_dir: str = "audio") -> List[ProcessingResult]:
    """
    Process all VTT files in a directory.
    
    Args:
        audio_dir: Directory containing VTT files
        
    Returns:
        List of ProcessingResult for each episode
    """
    audio_path = Path(audio_dir)
    vtt_files = sorted(audio_path.glob("*.vtt"))
    
    print(f"Found {len(vtt_files)} VTT files to process")
    print("=" * 60)
    
    results = []
    
    for i, vtt_file in enumerate(vtt_files, 1):
        episode_id = vtt_file.stem.replace(".en", "")
        print(f"\n[{i}/{len(vtt_files)}] Processing: {episode_id}")
        
        result = process_episode(str(vtt_file), episode_id)
        results.append(result)
        
        status = "✅" if result.success else "❌"
        print(f"{status} Completed in {result.duration_seconds:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Total sentences: {sum(r.sentences_count for r in successful)}")
    print(f"Total topics: {sum(r.topics_count for r in successful)}")
    print(f"Total time: {sum(r.duration_seconds for r in results):.1f}s")
    
    if failed:
        print(f"\nFailed episodes:")
        for r in failed:
            print(f"  - {r.episode_id}: {r.error}")
    
    # Show final index stats
    stats = get_index_stats()
    print(f"\nTypesense index stats: {stats}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific VTT file
        vtt_path = sys.argv[1]
        print(f"Processing single episode: {vtt_path}")
        result = process_episode(vtt_path)
        print(f"\nResult: {'Success' if result.success else 'Failed'}")
        if result.error:
            print(f"Error: {result.error}")
    else:
        # Process all episodes
        process_all_episodes()
