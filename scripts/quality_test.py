#!/usr/bin/env python3
"""
Comprehensive quality testing for Echo Audio Browser search.
Tests for hallucinations, relevance, and user satisfaction.
"""

import sys
sys.path.insert(0, '.')

from app.services.typesense_client import client
from app.services.typesense_search_v3 import search_categorized


def verify_term_in_content(topic_id: str, query: str) -> bool:
    """Verify the query term actually appears in the topic's full_text."""
    try:
        topic = client.collections['topics'].documents[topic_id].retrieve()
        full_text = topic.get('full_text', '').lower()
        return query.lower() in full_text
    except:
        return False


def test_query(query: str, limit: int = 5) -> dict:
    """Test a single query and return results with validation."""
    results = search_categorized(query, limit=limit)
    
    about = results.get('about', [])
    mentions = results.get('mentions', [])
    
    validated_about = []
    hallucinations = 0
    
    for r in about:
        topic_id = r.get('topic_id', r.get('id', ''))
        is_valid = verify_term_in_content(topic_id, query)
        
        validated_about.append({
            'topic_id': topic_id,
            'summary': r.get('summary', '')[:80],
            'valid': is_valid,
            'episode': r.get('episode_id', '')
        })
        
        if not is_valid:
            hallucinations += 1
    
    return {
        'query': query,
        'about_count': len(about),
        'mentions_count': len(mentions),
        'validated_results': validated_about,
        'hallucinations': hallucinations,
        'precision': (len(about) - hallucinations) / len(about) if about else 1.0
    }


def run_test_suite():
    """Run comprehensive test suite."""
    
    test_queries = [
        # People
        ('Trump', 'Person - frequent mention'),
        ('Elon Musk', 'Person - tech leader'),
        ('Chamath', 'Person - podcast host'),
        
        # Companies
        ('Tesla', 'Company - known hallucination issue'),
        ('NVIDIA', 'Company - tech'),
        ('OpenAI', 'Company - AI'),
        
        # Topics
        ('investing', 'Topic - general'),
        ('AI agents', 'Topic - specific'),
        ('immigration', 'Topic - political'),
        ('crypto', 'Topic - finance'),
        
        # Events/Specific
        ('DeepSeek', 'Event - recent AI'),
        ('inauguration', 'Event - political'),
        
        # Abstract
        ('productivity', 'Abstract concept'),
        ('career advice', 'Abstract - advice'),
        
        # Edge cases
        ('xyznonexistent', 'Negative test - should find nothing'),
    ]
    
    print('=' * 70)
    print('ECHO AUDIO BROWSER - QUALITY TEST SUITE')
    print('=' * 70)
    print()
    
    total_queries = 0
    total_results = 0
    total_hallucinations = 0
    failed_queries = []
    
    for query, description in test_queries:
        result = test_query(query)
        total_queries += 1
        total_results += result['about_count']
        total_hallucinations += result['hallucinations']
        
        # Determine status
        if result['about_count'] == 0 and 'nonexistent' not in query:
            status = '⚠️  NO RESULTS'
            failed_queries.append((query, 'no results'))
        elif result['hallucinations'] > 0:
            status = f"❌ {result['hallucinations']} HALLUCINATION(S)"
            failed_queries.append((query, f"{result['hallucinations']} hallucinations"))
        else:
            status = '✅ PASS'
        
        print(f'{status} | "{query}" ({description})')
        print(f'    ABOUT: {result["about_count"]} | MENTIONS: {result["mentions_count"]} | Precision: {result["precision"]:.0%}')
        
        # Show first 2 results for non-empty queries
        for r in result['validated_results'][:2]:
            valid_mark = '✓' if r['valid'] else '✗'
            print(f'    [{valid_mark}] {r["summary"]}...')
        
        print()
    
    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Queries tested: {total_queries}')
    print(f'Total ABOUT results: {total_results}')
    print(f'Total hallucinations: {total_hallucinations}')
    print(f'Overall precision: {(total_results - total_hallucinations) / total_results:.1%}' if total_results > 0 else 'N/A')
    print()
    
    if failed_queries:
        print('FAILED QUERIES:')
        for q, reason in failed_queries:
            print(f'  - "{q}": {reason}')
    else:
        print('ALL QUERIES PASSED! ✅')
    
    print()
    
    # Qualitative assessment
    print('=' * 70)
    print('QUALITATIVE ASSESSMENT')
    print('=' * 70)
    
    precision = (total_results - total_hallucinations) / total_results if total_results > 0 else 0
    
    if precision >= 0.95 and len(failed_queries) <= 1:
        print('Rating: EXCELLENT')
        print('User satisfaction: HIGH')
        print('Recommendation: Ready for user testing')
    elif precision >= 0.85 and len(failed_queries) <= 3:
        print('Rating: GOOD')
        print('User satisfaction: MEDIUM-HIGH')
        print('Recommendation: Minor issues, acceptable for testing')
    elif precision >= 0.70:
        print('Rating: ACCEPTABLE')
        print('User satisfaction: MEDIUM')
        print('Recommendation: Some issues need attention')
    else:
        print('Rating: NEEDS WORK')
        print('User satisfaction: LOW')
        print('Recommendation: Fix hallucinations before user testing')
    
    return {
        'passed': len(failed_queries) == 0,
        'precision': precision,
        'hallucinations': total_hallucinations,
        'failed_queries': failed_queries
    }


if __name__ == '__main__':
    run_test_suite()
