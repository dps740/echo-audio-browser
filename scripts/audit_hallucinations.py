#!/usr/bin/env python3
"""
Audit all topics for hallucinated summaries.
Checks if entities in summaries actually appear in the transcript.
"""

import sys
sys.path.insert(0, '.')

from app.services.typesense_client import client
from app.services.topic_segmentation_v2 import validate_summary_against_text


def audit_all_topics():
    """Scan all topics and flag ones with hallucinated summaries."""
    
    # Get all topics (paginated)
    all_hits = []
    page = 1
    per_page = 100
    
    while True:
        results = client.collections['topics'].documents.search({
            'q': '*',
            'per_page': per_page,
            'page': page
        })
        all_hits.extend(results['hits'])
        if len(results['hits']) < per_page:
            break
        page += 1
    
    total = len(all_hits)
    hallucinated_count = 0
    issues = []
    
    print(f"Auditing {total} topics for hallucinations...\n")
    
    for hit in all_hits:
        doc = hit['document']
        topic_id = doc['id']
        summary = doc.get('summary', '')
        full_text = doc.get('full_text', '')
        
        if not summary or not full_text:
            continue
        
        is_valid, hallucinated = validate_summary_against_text(summary, full_text)
        
        if not is_valid:
            hallucinated_count += 1
            issues.append({
                'topic_id': topic_id,
                'summary': summary,
                'hallucinated': hallucinated,
                'start_ms': doc.get('start_ms', 0)
            })
            
            # Print immediately
            print(f"❌ {topic_id}")
            print(f"   Summary: {summary[:80]}...")
            print(f"   Hallucinated: {hallucinated}")
            print()
    
    # Summary
    print("=" * 60)
    print(f"AUDIT COMPLETE")
    print(f"Total topics: {total}")
    print(f"Topics with hallucinations: {hallucinated_count} ({100*hallucinated_count/total:.1f}%)")
    print("=" * 60)
    
    # Group by episode
    episodes = {}
    for issue in issues:
        ep_id = issue['topic_id'].rsplit('_', 1)[0]
        if ep_id not in episodes:
            episodes[ep_id] = []
        episodes[ep_id].append(issue)
    
    print(f"\nAffected episodes: {len(episodes)}")
    for ep_id, ep_issues in sorted(episodes.items()):
        print(f"  {ep_id}: {len(ep_issues)} bad topics")
    
    return issues


if __name__ == "__main__":
    audit_all_topics()
