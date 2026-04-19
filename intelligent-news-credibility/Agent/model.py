"""
==============================================================================
 model.py — LangGraph Agentic Misinformation Monitoring Pipeline
==============================================================================

 PIPELINE OVERVIEW (3 Nodes):

   START
     │
     ▼
   [extract_claims_node]     ← Extracts factual claims from the article
     │
     ▼
   [retrieve_facts_node]     ← Queries ChromaDB for matching fact-checks
     │
     ▼
   [generate_assessment_node] ← Generates a final credibility report
     │
     ▼
   END

 KEY FEATURES:
   - Uses Groq API (Llama-3.1-8b) for structured claim extraction
   - Queries a persistent ChromaDB vector store built from the LIAR dataset
   - Threshold-based retrieval (score >= 0.5) prevents low-quality matches
   - Anti-hallucination prompting ensures the LLM never invents facts
   - Graceful degradation when no evidence is found in the database
==============================================================================
"""

import os
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


# ==============================================================================
# SECTION 1: Pydantic Schemas for Structured LLM Output
# ==============================================================================
# These schemas tell the LLM exactly what shape to return data in.
# LangChain's `with_structured_output()` uses these to parse the response.

class Claim(BaseModel):
    """Represents a single factual claim extracted from an article."""
    claim: str = Field(description="A single factual assertion extracted from the text")
    entity: str = Field(description="The main entity (person, org, place) involved in the claim")


class ClaimsOutput(BaseModel):
    """Container for all extracted claims — the LLM returns this structure."""
    extracted_claims: List[Claim]


# ==============================================================================
# SECTION 2: LLM Initialization (Groq API — Llama 3.1 8B)
# ==============================================================================

# --- Base LLM for general text generation (used in assessment node) ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,                          # Deterministic output
    api_key=os.getenv("GROQ_API_KEY"),
)

# --- Structured LLM for claim extraction (returns Pydantic objects) ---
# This wraps the same model but forces it to output valid ClaimsOutput JSON.
structured_llm = llm.with_structured_output(ClaimsOutput)


# ==============================================================================
# SECTION 3: Tavily Web Search Tool Initialization
# ==============================================================================

# Uses TAVILY_API_KEY from environment internally
retriever = TavilySearchResults(max_results=3)


# ==============================================================================
# SECTION 4: Graph State Definition
# ==============================================================================

class AgentState(TypedDict):
    article_text: str                          # Raw input article text
    extracted_claims: List[Claim]               # Claims extracted by Node 1
    retrieval_results: Dict[str, str]           # Claim → Evidence mapping from Node 2
    final_report: str                           # Credibility report from Node 3


# ==============================================================================
# SECTION 5: Node 1 — Claim Extraction
# ==============================================================================

def extract_claims_node(state: AgentState) -> AgentState:
    article_text = state["article_text"]

    # --- Prompt: Instruct the LLM to act as a fact-checking assistant ---
    prompt = """
You are an expert investigative journalist and fact-checking assistant.

Your task is to extract discrete, verifiable factual claims from the given article.

RULES:
- Extract ONLY factual assertions (statistics, events, actions, quotes)
- IGNORE opinions, predictions, or commentary
- Break complex sentences into atomic claims
- If no claims exist, return an empty list
"""

    try:
        # Invoke the structured LLM — it returns a ClaimsOutput Pydantic object
        result = structured_llm.invoke(
            prompt + f"\n\nArticle:\n{article_text}"
        )

        print(f"\n🔍 Extracted {len(result.extracted_claims)} claims from the article.")
        return {
            "extracted_claims": result.extracted_claims
        }

    except Exception as e:
        # --- Graceful error handling ---
        print(f"❌ Error in extract_claims_node: {e}")
        return {
            "extracted_claims": []
        }


# ==============================================================================
# SECTION 6: Node 2 — Fact Retrieval from ChromaDB (with Graceful Degradation)
# ==============================================================================

# --- Sentinel string for unverified claims ---
# This exact string is checked by the assessment prompt to prevent hallucination.
NO_EVIDENCE_SENTINEL = "NO VERIFIED EVIDENCE FOUND IN WEB SEARCH."


def retrieve_facts_node(state: AgentState) -> AgentState:
    """
    NODE 2: Query the ChromaDB vector store for each extracted claim.

    HOW IT WORKS:
      1. Iterates through each claim from Node 1.
      2. For each claim, queries the threshold retriever.
      3. TWO POSSIBLE OUTCOMES per claim:
         a) MATCH FOUND (docs returned):
            - Combines the page_content of all matching documents
            - Stores them as the evidence string for this claim
         b) NO MATCH (empty list returned):
            - The similarity score of all documents was below 0.5
            - We map this claim to the sentinel string:
              "NO VERIFIED EVIDENCE FOUND IN WEB SEARCH."
            - This is CRITICAL: it tells Node 3 NOT to hallucinate.

    WHY THIS MATTERS:
      Without this threshold + sentinel pattern, the LLM would try to
      "helpfully" verify claims using its training data — which could
      be outdated, incorrect, or completely fabricated. By explicitly
      flagging missing evidence, we force the LLM to admit uncertainty.
    """
    claims = state.get("extracted_claims", [])
    retrieval_results: Dict[str, str] = {}

    if not claims:
        print("⚠️  No claims to retrieve evidence for.")
        return {"retrieval_results": {}}

    print(f"\n📚 Retrieving evidence for {len(claims)} claims from ChromaDB...")

    for i, claim_obj in enumerate(claims):
        claim_text = claim_obj.claim
        print(f"\n   🔎 [{i+1}/{len(claims)}] Querying: \"{claim_text[:80]}...\"")

        try:
            # --- Query the threshold retriever ---
            # This returns ONLY documents with similarity >= 0.5.
            # If nothing qualifies, it returns an EMPTY list.
            matched_docs = retriever.invoke(claim_text)

            if matched_docs:
                # --- MATCH FOUND: Combine all matched documents ---
                print(f"      ✅ Found {len(matched_docs)} relevant fact-check(s).")

                # Join all matched documents' content with a separator
                combined_evidence = "\n---\n".join(
                    doc.page_content for doc in matched_docs
                )
                retrieval_results[claim_text] = combined_evidence
            else:
                # --- NO MATCH: Use the sentinel string ---
                print(f"      ⚠️  No evidence found above threshold (0.5).")
                retrieval_results[claim_text] = NO_EVIDENCE_SENTINEL

        except Exception as e:
            # --- Error during retrieval: treat as no evidence ---
            print(f"      ❌ Retrieval error: {e}")
            retrieval_results[claim_text] = NO_EVIDENCE_SENTINEL

    return {"retrieval_results": retrieval_results}


# ==============================================================================
# SECTION 7: Node 3 — Credibility Assessment (Anti-Hallucination Prompting)
# ==============================================================================

def generate_assessment_node(state: AgentState) -> AgentState:
    """
    NODE 3: Generate a final credibility assessment report.

    HOW IT WORKS:
      1. Reads the claims and their retrieval results from the state.
      2. Formats them into a structured context block.
      3. Sends everything to the LLM with a STRICT anti-hallucination prompt.
      4. The LLM generates a formatted report covering each claim.

    ANTI-HALLUCINATION STRATEGY:
      The system prompt contains an explicit instruction:
        "If the retrieved evidence says 'NO VERIFIED EVIDENCE FOUND IN WEB SEARCH.',
         you MUST state that the claim is 'Unverified due to lack of web evidence'."

      This prevents the LLM from:
        - Inventing fake sources or citations
        - Using its training data to "verify" claims (which may be wrong)
        - Giving false confidence on unverifiable statements
    """
    claims = state.get("extracted_claims", [])
    retrieval_results = state.get("retrieval_results", {})

    # --- Handle edge case: no claims were extracted ---
    if not claims:
        return {
            "final_report": "⚠️ No factual claims were extracted from this article. "
                            "The article may contain only opinions or commentary."
        }

    # --- Build the context block: pair each claim with its evidence ---
    context_parts = []
    for i, claim_obj in enumerate(claims):
        claim_text = claim_obj.claim
        evidence = retrieval_results.get(claim_text, NO_EVIDENCE_SENTINEL)
        context_parts.append(
            f"CLAIM {i+1}: \"{claim_text}\"\n"
            f"RETRIEVED EVIDENCE:\n{evidence}"
        )

    # Join all claim-evidence pairs into one context string
    full_context = "\n\n" + "=" * 40 + "\n\n".join(context_parts)

    # --- Anti-Hallucination System Prompt ---
    # This is the most critical part of the entire pipeline.
    # The prompt MUST be strict enough to prevent the LLM from inventing facts.
    system_prompt = """You are a rigorous fact-checking analyst producing a credibility report.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. For each claim, analyze ONLY the retrieved evidence provided below.
2. If the retrieved evidence for a claim says 'NO VERIFIED EVIDENCE FOUND IN WEB SEARCH.', 
   you MUST state in your Verdict that the claim is 'Unverified due to lack of web evidence'. 
   DO NOT invent facts, guess, or use baseline knowledge to verify the claim.
3. DO NOT fabricate sources, citations, URLs, or fact-check results.
4. If evidence IS found, compare the claim against the evidence and give your verdict 
   based SOLELY on what the evidence says.
5. Be transparent about the limitations of the evidence.

OUTPUT FORMAT:
For each claim, provide:
  - Claim: [the claim text]
  - Evidence Found: [Yes / No]
  - Verdict: [Your assessment based strictly on the evidence]
  - Confidence: [High / Medium / Low]
  - Reasoning: [Brief explanation of your verdict]

End with an OVERALL CREDIBILITY ASSESSMENT of the article.
"""

    # --- Build the user message with all claims and evidence ---
    user_message = f"""
Analyze the following claims extracted from a news article. 
For each claim, I have retrieved relevant fact-checks from a verified database.

{full_context}

Generate a detailed credibility report following the format specified.
"""

    try:
        # --- Invoke the LLM to generate the final report ---
        print("\n📝 Generating credibility assessment report...")
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])

        report = response.content
        print("   ✅ Report generated successfully.")
        return {"final_report": report}

    except Exception as e:
        print(f"❌ Error in generate_assessment_node: {e}")
        return {
            "final_report": f"❌ Failed to generate assessment report. Error: {str(e)}"
        }


# ==============================================================================
# SECTION 8: Build the LangGraph Pipeline
# ==============================================================================
# The graph connects the three nodes in sequence:
#   START → extract_claims → retrieve_facts → generate_assessment → END

graph = StateGraph(AgentState)

# --- Register all three nodes ---
graph.add_node("extract_claims_node", extract_claims_node)
graph.add_node("retrieve_facts_node", retrieve_facts_node)
graph.add_node("generate_assessment_node", generate_assessment_node)

# --- Wire the edges (sequential pipeline) ---
graph.add_edge(START, "extract_claims_node")
graph.add_edge("extract_claims_node", "retrieve_facts_node")
graph.add_edge("retrieve_facts_node", "generate_assessment_node")
graph.add_edge("generate_assessment_node", END)

# --- Compile the graph into a runnable workflow ---
workflow = graph.compile()
