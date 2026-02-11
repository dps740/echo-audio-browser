#!/usr/bin/env python3
"""
Re-segment episodes with improved validation.
Can process a single episode or all episodes.

Usage:
    python3 scripts/resegment_episodes.py              # All episodes
    python3 scripts/resegment_episodes.py gXY1kx7zlkk  # Single episode
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

print("Script starting...", flush=True)
sys.path.insert(0, '.')

print("Importing typesense client...", flush=True)
from app.services.typesense_client import client
print("Importing topic segmentation...", flush=True)
from app.services.topic_segmentation_v2 import segment_sentences
print("Imports complete.", flush=True)


@dataclass  
class Sentence:
    """A sentence with timestamp and extracted entities (from Typesense)."""
    text: str
    start_ms: int
    end_ms: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


def get_all_episodes():
    """Get list of all episodes currently indexed."""
    results = client.collections['topics'].documents.search({
        'q': '*',
        'per_page': 250,
        'group_by': 'episode_id',
        'group_limit': 1
    })
    
    episodes = set()
    for group in results.get('grouped_hits', []):
        if group.get('hits'):
            ep_id = group['hits'][0]['document'].get('episode_id')
            if ep_id:
                episodes.add(ep_id)
    
    return sorted(episodes)


def fetch_sentences_from_typesense(episode_id: str) -> List[Sentence]:
    """Fetch existing sentences from Typesense index."""
    all_sentences = []
    page = 1
    per_page = 250
    
    while True:
        results = client.collections['sentences'].documents.search({
            'q': '*',
            'filter_by': f'episode_id:={episode_id}',
            'sort_by': 'start_ms:asc',
            'per_page': per_page,
            'page': page
        })
        
        for hit in results['hits']:
            doc = hit['document']
            all_sentences.append(Sentence(
                text=doc.get('text', ''),
                start_ms=doc.get('start_ms', 0),
                end_ms=doc.get('end_ms', 0),
                people=doc.get('people', []),
                companies=doc.get('companies', []),
                topics=doc.get('topics', [])
            ))
        
        if len(results['hits']) < per_page:
            break
        page += 1
    
    return all_sentences


def resegment_episode(episode_id: str, audio_dir: str = "audio"):
    """Re-segment a single episode with validation."""
    
    print(f"  📥 Fetching sentences from Typesense...", flush=True)
    sentences = fetch_sentences_from_typesense(episode_id)
    
    if not sentences:
        print(f"  ❌ No sentences found for {episode_id}", flush=True)
        return False
    
    print(f"  📊 Found {len(sentences)} sentences", flush=True)
    
    print(f"  🤖 Segmenting with LLM (includes validation)...", flush=True)
    topics = segment_sentences(sentences, episode_id)
    print(f"  ✅ Created {len(topics)} topics", flush=True)
    
    # Delete old topics for this episode
    print(f"  🗑️  Deleting old topics...", flush=True)
    try:
        old_results = client.collections['topics'].documents.search({
            'q': '*',
            'filter_by': f'episode_id:={episode_id}',
            'per_page': 100
        })
        for hit in old_results['hits']:
            try:
                client.collections['topics'].documents[hit['document']['id']].delete()
            except:
                pass
    except Exception as e:
        print(f"  ⚠️  Error deleting old topics: {e}", flush=True)
    
    # Index new topics
    print(f"  📥 Indexing new topics...", flush=True)
    for topic in topics:
        doc = {
            'id': topic.topic_id,
            'episode_id': topic.episode_id,
            'summary': topic.summary,
            'full_text': topic.full_text,
            'start_ms': topic.start_ms,
            'end_ms': topic.end_ms,
            'sentence_start_idx': topic.sentence_start_idx,
            'sentence_end_idx': topic.sentence_end_idx,
            'people': topic.people,
            'companies': topic.companies,
            'keywords': topic.keywords,
        }
        try:
            client.collections['topics'].documents.upsert(doc)
        except Exception as e:
            print(f"  ⚠️  Error indexing {topic.topic_id}: {e}", flush=True)
    
    # Update sentence topic_ids
    print(f"  🔗 Updating sentence topic mappings...", flush=True)
    for topic in topics:
        for sent_idx in range(topic.sentence_start_idx, topic.sentence_end_idx + 1):
            sent_id = f"{episode_id}_s{sent_idx}"
            try:
                client.collections['sentences'].documents[sent_id].update({
                    'topic_id': topic.topic_id
                })
            except:
                pass  # Sentence might not exist
    
    print(f"  ✅ Done!", flush=True)
    return True


def main():
    if len(sys.argv) > 1:
        # Single episode
        episode_id = sys.argv[1]
        print(f"Re-segmenting episode: {episode_id}", flush=True)
        resegment_episode(episode_id)
    else:
        # All episodes
        episodes = get_all_episodes()
        print(f"Re-segmenting {len(episodes)} episodes...\n", flush=True)
        
        for i, episode_id in enumerate(episodes):
            print(f"\n[{i+1}/{len(episodes)}] {episode_id}", flush=True)
            try:
                resegment_episode(episode_id)
            except Exception as e:
                print(f"  ❌ Error: {e}", flush=True)
        
        print(f"\n✅ Complete! Re-segmented {len(episodes)} episodes.", flush=True)


if __name__ == "__main__":
    main()
