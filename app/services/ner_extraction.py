"""
Named Entity Recognition for podcast transcripts using spaCy.
Extracts people, companies, and topic keywords from text.
"""

import spacy
from typing import List, Dict, Set
from dataclasses import dataclass, field

# Load spaCy model (small English model, fast)
nlp = spacy.load("en_core_web_sm")


@dataclass
class ExtractedEntities:
    """Entities extracted from text."""
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


def extract_entities(text: str) -> ExtractedEntities:
    """
    Extract named entities from text using spaCy.
    
    Returns:
        ExtractedEntities with people, companies, and topics
    """
    doc = nlp(text)
    
    people: Set[str] = set()
    companies: Set[str] = set()
    topics: Set[str] = set()
    
    for ent in doc.ents:
        # Normalize entity text
        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue
            
        if ent.label_ == "PERSON":
            people.add(entity_text)
        elif ent.label_ in ("ORG", "COMPANY"):
            companies.add(entity_text)
        elif ent.label_ in ("PRODUCT", "WORK_OF_ART", "EVENT", "LAW"):
            # These often represent discussable topics
            topics.add(entity_text)
        elif ent.label_ in ("GPE", "LOC", "FAC"):
            # Geopolitical entities, locations - can be topics
            topics.add(entity_text)
        elif ent.label_ == "NORP":
            # Nationalities, religious, political groups
            topics.add(entity_text)
    
    return ExtractedEntities(
        people=sorted(list(people)),
        companies=sorted(list(companies)),
        topics=sorted(list(topics))
    )


def extract_entities_batch(texts: List[str]) -> List[ExtractedEntities]:
    """
    Extract entities from multiple texts efficiently using spaCy's pipe.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of ExtractedEntities, one per input text
    """
    results = []
    
    # Process in batches using spaCy's efficient pipe
    for doc in nlp.pipe(texts, batch_size=50):
        people: Set[str] = set()
        companies: Set[str] = set()
        topics: Set[str] = set()
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) < 2:
                continue
                
            if ent.label_ == "PERSON":
                people.add(entity_text)
            elif ent.label_ in ("ORG", "COMPANY"):
                companies.add(entity_text)
            elif ent.label_ in ("PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "GPE", "LOC", "FAC", "NORP"):
                topics.add(entity_text)
        
        results.append(ExtractedEntities(
            people=sorted(list(people)),
            companies=sorted(list(companies)),
            topics=sorted(list(topics))
        ))
    
    return results


def merge_entities(entities_list: List[ExtractedEntities]) -> ExtractedEntities:
    """
    Merge multiple ExtractedEntities into one (for aggregating across sentences).
    """
    all_people: Set[str] = set()
    all_companies: Set[str] = set()
    all_topics: Set[str] = set()
    
    for entities in entities_list:
        all_people.update(entities.people)
        all_companies.update(entities.companies)
        all_topics.update(entities.topics)
    
    return ExtractedEntities(
        people=sorted(list(all_people)),
        companies=sorted(list(all_companies)),
        topics=sorted(list(all_topics))
    )


if __name__ == "__main__":
    # Test with sample podcast transcript
    test_text = """
    In this episode, Elon Musk discusses Tesla's new AI initiatives with Sam Altman.
    They talk about OpenAI's ChatGPT and how it compares to Google's Bard.
    The conversation touches on the future of Silicon Valley and investments in 
    artificial intelligence startups. Musk mentions SpaceX's Starship program
    and potential collaboration with NASA on Mars missions.
    """
    
    print("Testing NER extraction...")
    print("-" * 50)
    
    entities = extract_entities(test_text)
    
    print(f"People: {entities.people}")
    print(f"Companies: {entities.companies}")
    print(f"Topics: {entities.topics}")
    
    # Test batch extraction
    print("\nTesting batch extraction...")
    sentences = [
        "Elon Musk founded Tesla and SpaceX.",
        "Sam Altman leads OpenAI in San Francisco.",
        "Apple and Microsoft are competing in AI."
    ]
    
    batch_results = extract_entities_batch(sentences)
    for i, (sent, ents) in enumerate(zip(sentences, batch_results)):
        print(f"\n{i+1}. '{sent[:50]}...'")
        print(f"   People: {ents.people}, Companies: {ents.companies}")
