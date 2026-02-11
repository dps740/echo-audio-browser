"""
Typesense client and schema management for Echo Audio Browser.
"""

import typesense
from app.config import settings

# Initialize Typesense client
client = typesense.Client({
    'api_key': settings.typesense_api_key,
    'nodes': [{
        'host': settings.typesense_host,
        'port': settings.typesense_port,
        'protocol': 'http'
    }],
    'connection_timeout_seconds': 10
})

# Schema definitions
SENTENCES_SCHEMA = {
    'name': 'sentences',
    'fields': [
        {'name': 'episode_id', 'type': 'string', 'facet': True},
        {'name': 'topic_id', 'type': 'string', 'facet': True},
        {'name': 'text', 'type': 'string'},
        {'name': 'start_ms', 'type': 'int64'},
        {'name': 'end_ms', 'type': 'int64'},
        {'name': 'people', 'type': 'string[]', 'facet': True},
        {'name': 'companies', 'type': 'string[]', 'facet': True},
        {'name': 'topics', 'type': 'string[]', 'facet': True},
    ],
    'default_sorting_field': 'start_ms'
}

TOPICS_SCHEMA = {
    'name': 'topics',
    'fields': [
        {'name': 'episode_id', 'type': 'string', 'facet': True},
        {'name': 'summary', 'type': 'string'},
        {'name': 'full_text', 'type': 'string'},
        {'name': 'start_ms', 'type': 'int64'},
        {'name': 'end_ms', 'type': 'int64'},
        {'name': 'people', 'type': 'string[]', 'facet': True},
        {'name': 'companies', 'type': 'string[]', 'facet': True},
        {'name': 'keywords', 'type': 'string[]', 'facet': True},
    ],
    'default_sorting_field': 'start_ms'
}


def create_collections():
    """Create or recreate the Typesense collections."""
    # Delete existing collections if they exist
    for name in ['sentences', 'topics']:
        try:
            client.collections[name].delete()
            print(f"Deleted existing '{name}' collection")
        except typesense.exceptions.ObjectNotFound:
            pass
    
    # Create fresh collections
    client.collections.create(SENTENCES_SCHEMA)
    print("Created 'sentences' collection")
    
    client.collections.create(TOPICS_SCHEMA)
    print("Created 'topics' collection")


def get_collections_info():
    """Get info about existing collections."""
    try:
        sentences = client.collections['sentences'].retrieve()
        topics = client.collections['topics'].retrieve()
        return {
            'sentences': {
                'num_documents': sentences.get('num_documents', 0),
                'fields': len(sentences.get('fields', []))
            },
            'topics': {
                'num_documents': topics.get('num_documents', 0),
                'fields': len(topics.get('fields', []))
            }
        }
    except typesense.exceptions.ObjectNotFound as e:
        return {'error': str(e)}


def health_check():
    """Check Typesense health."""
    try:
        health = client.operations.is_healthy()
        return {'healthy': health}
    except Exception as e:
        return {'healthy': False, 'error': str(e)}


if __name__ == '__main__':
    print("Health:", health_check())
    print("\nCreating collections...")
    create_collections()
    print("\nCollections info:", get_collections_info())
