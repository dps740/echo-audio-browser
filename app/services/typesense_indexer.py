"""
Index sentences and topics into Typesense.
"""

from typing import List
import typesense

from app.services.typesense_client import client
from app.services.sentence_parser import Sentence
from app.services.topic_segmentation_v2 import Topic


def index_sentences(sentences: List[Sentence], episode_id: str, topics: List[Topic]) -> int:
    """
    Index sentences into Typesense.
    
    Args:
        sentences: List of parsed sentences
        episode_id: Episode identifier
        topics: List of topics (to get topic_id for each sentence)
        
    Returns:
        Number of sentences indexed
    """
    # Build sentence -> topic_id mapping
    sentence_to_topic = {}
    for topic in topics:
        for i in range(topic.sentence_start_idx, topic.sentence_end_idx + 1):
            sentence_to_topic[i] = topic.topic_id
    
    # Build documents
    documents = []
    for i, sent in enumerate(sentences):
        topic_id = sentence_to_topic.get(i, topics[-1].topic_id if topics else "unknown")
        
        doc = {
            "id": f"{episode_id}_s{i}",
            "episode_id": episode_id,
            "topic_id": topic_id,
            "text": sent.text.replace('&gt;', '>').replace('&lt;', '<'),
            "start_ms": sent.start_ms,
            "end_ms": sent.end_ms,
            "people": sent.people if sent.people else [""],  # Typesense needs non-empty arrays
            "companies": sent.companies if sent.companies else [""],
            "topics": sent.topics if sent.topics else [""]
        }
        documents.append(doc)
    
    # Batch import
    if documents:
        # Import in batches of 100
        for i in range(0, len(documents), 100):
            batch = documents[i:i+100]
            try:
                result = client.collections['sentences'].documents.import_(batch, {'action': 'upsert'})
                # Check for errors
                errors = [r for r in result if not r.get('success', True)]
                if errors:
                    print(f"  Warning: {len(errors)} sentence import errors")
            except Exception as e:
                print(f"  Error importing sentences batch {i}: {e}")
    
    return len(documents)


def index_topics(topics: List[Topic]) -> int:
    """
    Index topics into Typesense.
    
    Args:
        topics: List of topic segments
        
    Returns:
        Number of topics indexed
    """
    documents = []
    for topic in topics:
        doc = {
            "id": topic.topic_id,
            "episode_id": topic.episode_id,
            "summary": topic.summary,
            "full_text": topic.full_text[:50000],  # Limit size
            "start_ms": topic.start_ms,
            "end_ms": topic.end_ms,
            "people": topic.people if topic.people else [""],
            "companies": topic.companies if topic.companies else [""],
            "keywords": topic.keywords if topic.keywords else [""]
        }
        documents.append(doc)
    
    # Import all topics
    if documents:
        try:
            result = client.collections['topics'].documents.import_(documents, {'action': 'upsert'})
            errors = [r for r in result if not r.get('success', True)]
            if errors:
                print(f"  Warning: {len(errors)} topic import errors")
        except Exception as e:
            print(f"  Error importing topics: {e}")
    
    return len(documents)


def delete_episode(episode_id: str) -> dict:
    """
    Delete all sentences and topics for an episode.
    
    Args:
        episode_id: Episode identifier
        
    Returns:
        Dict with deletion counts
    """
    result = {'sentences': 0, 'topics': 0}
    
    try:
        # Delete sentences
        del_result = client.collections['sentences'].documents.delete({
            'filter_by': f'episode_id:={episode_id}'
        })
        result['sentences'] = del_result.get('num_deleted', 0)
    except Exception as e:
        print(f"Error deleting sentences for {episode_id}: {e}")
    
    try:
        # Delete topics
        del_result = client.collections['topics'].documents.delete({
            'filter_by': f'episode_id:={episode_id}'
        })
        result['topics'] = del_result.get('num_deleted', 0)
    except Exception as e:
        print(f"Error deleting topics for {episode_id}: {e}")
    
    return result


def get_index_stats() -> dict:
    """Get statistics about indexed content."""
    try:
        sentences_info = client.collections['sentences'].retrieve()
        topics_info = client.collections['topics'].retrieve()
        
        return {
            'sentences': sentences_info.get('num_documents', 0),
            'topics': topics_info.get('num_documents', 0)
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    print("Index stats:", get_index_stats())
