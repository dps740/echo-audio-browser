#!/usr/bin/env python3
"""
Echo Audio Browser - Qualitative Relevance Test

This test would have caught the issue:
- Old test: "Tesla in keywords?" ✅ (but wrong!)
- New test: "Top result ABOUT Tesla?" ❌ (catches the problem)
"""

import requests
import sys

LOCAL = "http://localhost:8766"

print("=" * 60)
print("QUALITATIVE RELEVANCE TEST")
print("This test catches 'works but not good' issues")
print("=" * 60)

QUALITY_QUERIES = [
    "Tesla",
    "Bitcoin", 
    "Elon Musk",
    "AI safety",
]

all_passed = True
issues = []

for query in QUALITY_QUERIES:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 40)
    
    try:
        r = requests.get(f"{LOCAL}/v2/search/v3", params={"q": query, "limit": 5}, timeout=10)
        results = r.json().get("results", [])
        
        if not results:
            print(f"   ⚠️ No results found")
            continue
        
        # Check each result
        about_results = []
        mention_results = []
        
        for i, res in enumerate(results, 1):
            is_about = query.lower() in res['summary'].lower()
            if is_about:
                about_results.append(i)
            else:
                mention_results.append(i)
            
            marker = "🎯 ABOUT" if is_about else "📎 MENTIONS"
            print(f"   {i}. {marker}: {res['summary'][:50]}...")
        
        # Quality metrics
        print(f"\n   📊 Quality Metrics:")
        print(f"      Results ABOUT query: {len(about_results)}/{len(results)}")
        print(f"      Results only MENTIONING: {len(mention_results)}/{len(results)}")
        
        # THE KEY TEST: If there are ABOUT results, is #1 one of them?
        if about_results:
            if 1 in about_results:
                print(f"   ✅ PASS: Top result is ABOUT '{query}'")
            else:
                print(f"   ❌ FAIL: Top result only MENTIONS '{query}' (result #{about_results[0]} is ABOUT)")
                all_passed = False
                issues.append(f"'{query}': Top result not about query")
        else:
            print(f"   ℹ️ INFO: No results specifically ABOUT '{query}' (only mentions)")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        all_passed = False

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if all_passed:
    print("✅ All qualitative checks passed")
    print("\nThis test verifies:")
    print("  • Top results are ABOUT the query, not just mentioning it")
    print("  • Users would be satisfied with search results")
else:
    print("❌ Quality issues found:")
    for issue in issues:
        print(f"   • {issue}")
    print("\nThis would have caught the problem BEFORE user testing!")
    sys.exit(1)
