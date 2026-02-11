#!/usr/bin/env python3
"""
Comprehensive Level 3 Search Test Suite

Tests:
1. Problem queries (space, currency) - the issues we fixed
2. Entity queries (people, companies)
3. Topic queries (abstract concepts)
4. Edge cases (short queries, phrases)
5. Comparison with V3 search
6. Quality metrics
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.search_l3 import search_l3
from app.services.typesense_search_v3 import search_categorized as search_v3


@dataclass
class TestCase:
    query: str
    description: str
    expected_about: List[str]  # Keywords that SHOULD appear in ABOUT results
    not_expected: List[str]    # Keywords that should NOT appear (false positives)
    min_about_count: int = 1


# Define test cases
TEST_CASES = [
    # === PROBLEM QUERIES (the ones we fixed) ===
    TestCase(
        query="space",
        description="Polysemy test - should return aerospace, not 'crypto space'",
        expected_about=["black hole", "dark energy", "cosmolog", "physics", "universe", "SpaceX", "NASA", "rocket", "satellite"],
        not_expected=["crypto space", "AI space", "industry space"],
        min_about_count=1
    ),
    TestCase(
        query="currency",
        description="Synonym test - should find fiat/stablecoin content",
        expected_about=["fiat", "stablecoin", "money", "dollar", "monetary", "crypto"],
        not_expected=[],
        min_about_count=1
    ),
    
    # === ENTITY QUERIES ===
    TestCase(
        query="Elon Musk",
        description="Person query",
        expected_about=["Elon", "Musk", "Tesla", "SpaceX", "X"],
        not_expected=[],
        min_about_count=1
    ),
    TestCase(
        query="OpenAI",
        description="Company query",
        expected_about=["OpenAI", "GPT", "ChatGPT", "Sam Altman", "AI"],
        not_expected=[],
        min_about_count=1
    ),
    TestCase(
        query="Tesla",
        description="Company query - should find Tesla the company",
        expected_about=["Tesla", "electric", "Elon", "car", "vehicle"],
        not_expected=[],
        min_about_count=1
    ),
    
    # === TOPIC QUERIES ===
    TestCase(
        query="artificial intelligence",
        description="Topic query with phrase",
        expected_about=["AI", "artificial intelligence", "machine learning", "neural", "GPT", "model"],
        not_expected=[],
        min_about_count=1
    ),
    TestCase(
        query="Bitcoin",
        description="Crypto topic",
        expected_about=["Bitcoin", "BTC", "crypto", "blockchain"],
        not_expected=[],
        min_about_count=1
    ),
    TestCase(
        query="climate",
        description="Abstract topic",
        expected_about=["climate", "warming", "carbon", "environment", "green"],
        not_expected=[],
        min_about_count=0  # May not have content
    ),
    
    # === EDGE CASES ===
    TestCase(
        query="AI",
        description="Short acronym query",
        expected_about=["AI", "artificial intelligence", "machine learning"],
        not_expected=[],
        min_about_count=1
    ),
    TestCase(
        query="future of work",
        description="Phrase query",
        expected_about=["work", "job", "remote", "automation", "career"],
        not_expected=[],
        min_about_count=0
    ),
]


def run_single_test(test: TestCase, verbose: bool = True) -> Dict[str, Any]:
    """Run a single test case and return results."""
    start_time = time.time()
    
    try:
        results = search_l3(test.query, limit=10, min_confidence="low")
        elapsed = time.time() - start_time
        
        about_results = results.get("about", [])
        mentions_results = results.get("mentions", [])
        related_results = results.get("related", [])
        
        # Check for expected keywords in ABOUT results
        about_text = " ".join([r.get("summary", "") + " " + r.get("explanation", "") 
                               for r in about_results]).lower()
        
        found_expected = []
        missing_expected = []
        for keyword in test.expected_about:
            if keyword.lower() in about_text:
                found_expected.append(keyword)
            else:
                missing_expected.append(keyword)
        
        # Check for unexpected keywords (false positives)
        found_unexpected = []
        for keyword in test.not_expected:
            if keyword.lower() in about_text:
                found_unexpected.append(keyword)
        
        # Calculate scores
        has_min_results = len(about_results) >= test.min_about_count
        expected_ratio = len(found_expected) / len(test.expected_about) if test.expected_about else 1.0
        no_false_positives = len(found_unexpected) == 0
        
        passed = has_min_results and expected_ratio >= 0.3 and no_false_positives
        
        result = {
            "query": test.query,
            "description": test.description,
            "passed": passed,
            "elapsed_seconds": round(elapsed, 2),
            "about_count": len(about_results),
            "mentions_count": len(mentions_results),
            "related_count": len(related_results),
            "expanded_terms": results.get("expanded_terms", []),
            "intent": results.get("intent", ""),
            "found_expected": found_expected,
            "missing_expected": missing_expected,
            "found_unexpected": found_unexpected,
            "has_min_results": has_min_results,
            "expected_ratio": round(expected_ratio, 2),
            "top_results": [
                {
                    "summary": r.get("summary", "")[:60],
                    "confidence": r.get("confidence", ""),
                    "score": r.get("relevance_score", 0)
                }
                for r in about_results[:3]
            ]
        }
        
        if verbose:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} | {test.query}")
            print(f"  Description: {test.description}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  ABOUT: {len(about_results)} | MENTIONS: {len(mentions_results)} | RELATED: {len(related_results)}")
            print(f"  Expanded: {results.get('expanded_terms', [])[:5]}")
            if about_results:
                print(f"  Top result: {about_results[0].get('summary', '')[:60]}...")
            if missing_expected:
                print(f"  Missing keywords: {missing_expected}")
            if found_unexpected:
                print(f"  ⚠️  False positives: {found_unexpected}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n❌ ERROR | {test.query}: {e}")
        return {
            "query": test.query,
            "description": test.description,
            "passed": False,
            "error": str(e)
        }


def compare_with_v3(query: str) -> Dict[str, Any]:
    """Compare L3 results with V3 results for a query."""
    print(f"\n=== Comparing L3 vs V3 for '{query}' ===")
    
    # Run V3 search
    v3_start = time.time()
    v3_results = search_v3(query, limit=10)
    v3_time = time.time() - v3_start
    
    # Run L3 search
    l3_start = time.time()
    l3_results = search_l3(query, limit=10)
    l3_time = time.time() - l3_start
    
    print(f"\nV3 (BM25 only):")
    print(f"  Time: {v3_time:.2f}s")
    print(f"  ABOUT: {v3_results.get('about_count', 0)}")
    for r in v3_results.get('about', [])[:3]:
        print(f"    - {r.get('summary', '')[:50]}...")
    
    print(f"\nL3 (Hybrid + Rerank):")
    print(f"  Time: {l3_time:.2f}s")
    print(f"  ABOUT: {l3_results.get('about_count', 0)}")
    print(f"  Expanded: {l3_results.get('expanded_terms', [])}")
    for r in l3_results.get('about', [])[:3]:
        print(f"    - [{r.get('confidence', '')}] {r.get('summary', '')[:50]}...")
        print(f"      Why: {r.get('explanation', '')[:60]}...")
    
    return {
        "query": query,
        "v3_about_count": v3_results.get('about_count', 0),
        "l3_about_count": l3_results.get('about_count', 0),
        "v3_time": round(v3_time, 2),
        "l3_time": round(l3_time, 2),
        "improvement": l3_results.get('about_count', 0) - v3_results.get('about_count', 0)
    }


def run_all_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run all test cases and return summary."""
    print("=" * 70)
    print("LEVEL 3 SEARCH TEST SUITE")
    print("=" * 70)
    
    results = []
    passed = 0
    failed = 0
    total_time = 0
    
    for test in TEST_CASES:
        result = run_single_test(test, verbose)
        results.append(result)
        if result.get("passed"):
            passed += 1
        else:
            failed += 1
        total_time += result.get("elapsed_seconds", 0)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print(f"Failed: {failed}/{len(TEST_CASES)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per query: {total_time/len(TEST_CASES):.1f}s")
    
    # Show failures
    failures = [r for r in results if not r.get("passed")]
    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f['query']}: {f.get('error', 'Did not meet expectations')}")
    
    return {
        "passed": passed,
        "failed": failed,
        "total": len(TEST_CASES),
        "pass_rate": round(passed / len(TEST_CASES) * 100, 1),
        "total_time_seconds": round(total_time, 1),
        "avg_time_seconds": round(total_time / len(TEST_CASES), 1),
        "results": results
    }


def run_comparison_tests() -> List[Dict]:
    """Run comparison tests between V3 and L3."""
    print("\n" + "=" * 70)
    print("V3 vs L3 COMPARISON")
    print("=" * 70)
    
    comparison_queries = ["space", "currency", "Tesla", "AI", "Bitcoin"]
    results = []
    
    for query in comparison_queries:
        result = compare_with_v3(query)
        results.append(result)
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Query':<15} {'V3 ABOUT':<10} {'L3 ABOUT':<10} {'Δ':<5} {'V3 Time':<10} {'L3 Time':<10}")
    print("-" * 70)
    for r in results:
        delta = r['improvement']
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{r['query']:<15} {r['v3_about_count']:<10} {r['l3_about_count']:<10} {delta_str:<5} {r['v3_time']:<10} {r['l3_time']:<10}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Level 3 Search")
    parser.add_argument("--compare", action="store_true", help="Run V3 vs L3 comparison")
    parser.add_argument("--quick", action="store_true", help="Run just the problem queries")
    parser.add_argument("--query", type=str, help="Test a specific query")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    
    args = parser.parse_args()
    
    if args.query:
        # Test single query
        result = search_l3(args.query, limit=10)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nQuery: {args.query}")
            print(f"Intent: {result.get('intent', '')}")
            print(f"Expanded: {result.get('expanded_terms', [])}")
            print(f"\nABOUT ({result.get('about_count', 0)}):")
            for r in result.get('about', []):
                print(f"  [{r.get('confidence', '')}] {r.get('summary', '')}")
                print(f"    Why: {r.get('explanation', '')}")
    
    elif args.compare:
        run_comparison_tests()
    
    elif args.quick:
        # Just the problem queries
        quick_tests = [t for t in TEST_CASES if t.query in ["space", "currency"]]
        for test in quick_tests:
            run_single_test(test, verbose=True)
    
    else:
        # Full test suite
        summary = run_all_tests(verbose=True)
        
        if args.json:
            print("\n" + json.dumps(summary, indent=2))
