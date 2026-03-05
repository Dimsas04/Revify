import pandas as pd
import json
import time
from datetime import datetime
from main import extract_json_from_markdown, summarize_reviews_chunked
from src.revify_flow.crews.team_revify.team_revify import TeamRevify
from crewai import Crew, Process
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm.exceptions import RateLimitError
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import os

# ─────────────────────────────────────────────
# CONFIG — only change things here
# ─────────────────────────────────────────────
REVIEWS_FILE = "benchmark_reviews_250.csv"
CHUNK_SIZE   = 175  # optimal chunk size from ablation study
OUTPUT_DIR   = "parts_ablation_output"
SCORES_CSV   = os.path.join(OUTPUT_DIR, "parts_ablation_scores.csv")

# VARIANT: one of "full", "no_summary", "no_feature_extraction"
# Change this per run
VARIANT = "no_feature_extraction"

FEATURES = [
    "Voice Assistant Performance (Alexa Integration)",
    "Display Quality",
    "Sound Quality",
    "Smart Home Control Capabilities",
    "Privacy and Security Features",
    "Camera Quality",
    "Ease of Setup and Use",
    "Design and Build Quality",
    "Connectivity (Wifi and Bluetooth)",
    "Value for Money"
]
# ─────────────────────────────────────────────


def compute_bert_score(hypothesis: str, reference: str) -> dict:
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {'bert_f1': round(F1.mean().item(), 4)}


def compute_rouge_scores(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1_f1': round(scores['rouge1'].fmeasure, 4),
        'rouge2_f1': round(scores['rouge2'].fmeasure, 4),
        'rougeL_f1': round(scores['rougeL'].fmeasure, 4),
    }


def append_to_csv(row: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(SCORES_CSV)
    df_row.to_csv(SCORES_CSV, mode='a', header=write_header, index=False)
    print(f"📊 Scores appended to: {SCORES_CSV}")


def extract_hypothesis(raw_output, analysis_results):
    """Extract verdict text from JSON output as hypothesis."""
    try:
        hypothesis_text = " ".join(
            item.get('verdict', '') + " " + item.get('sentiment', '')
            for item in analysis_results
            if isinstance(item, dict)
        ).strip()
        if not hypothesis_text:
            print("⚠️  Could not extract verdicts, falling back to raw output")
            return raw_output
        return hypothesis_text
    except Exception:
        return raw_output


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=15, max=120),
    stop=stop_after_attempt(5)
)
def run_full_revify(reviews_df):
    """
    VARIANT 1: Full Revify pipeline
    Scrape -> Chunk Summary Agent (chunk=175) -> Review Analysis Agent
    Reference: concatenated chunk summaries
    """
    print(f"\n{'='*60}")
    print(f"🔬 VARIANT: Full Revify  [chunk_size={CHUNK_SIZE}]")
    print(f"{'='*60}\n")

    team = TeamRevify()
    df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
    review_dicts = df_filtered.to_dict(orient='records')

    metrics = {
        'variant': 'full_revify',
        'total_reviews': len(review_dicts),
        'run_timestamp': datetime.now().isoformat(),
        'summarization_time': 0,
        'analysis_time': 0,
        'total_time': 0,
        'num_chunks': CHUNK_SIZE,
        'parsing_success': False,
    }

    start_total = time.time()

    # Phase 1: Chunked summarization
    print(f"📝 Summarising {len(review_dicts)} reviews [chunk_size={CHUNK_SIZE}]...")
    t0 = time.time()
    chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=CHUNK_SIZE)
    metrics['summarization_time'] = round(time.time() - t0, 2)
    metrics['num_chunks'] = len(chunk_summaries)

    reference_text = "\n\n".join(chunk_summaries)
    reviews_input  = reference_text
    print(f"✅ {len(chunk_summaries)} chunk summaries ({metrics['summarization_time']}s)")

    # Phase 2: Analysis
    print(f"\n🤖 Running Review Analysis Agent...")
    t0 = time.time()

    review_agent  = team.review_analysis_agent()
    analysis_task = team.comprehensive_review_analysis_task()
    analysis_crew = Crew(agents=[review_agent], tasks=[analysis_task],
                         process=Process.sequential, verbose=False)

    result = analysis_crew.kickoff(inputs={
        "features": ", ".join(FEATURES),
        "reviews":  reviews_input
    })

    metrics['analysis_time'] = round(time.time() - t0, 2)
    metrics['total_time']    = round(time.time() - start_total, 2)

    raw_output = result.raw
    json_str   = extract_json_from_markdown(raw_output)

    try:
        analysis_results = json.loads(json_str)
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results] if isinstance(analysis_results, dict) else []
        metrics['parsing_success'] = True
    except Exception as e:
        analysis_results = []
        metrics['parsing_success'] = False

    hypothesis_text = extract_hypothesis(raw_output, analysis_results)

    return metrics, reference_text, hypothesis_text, analysis_results


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=15, max=120),
    stop=stop_after_attempt(5)
)
def run_no_summary_module(reviews_df):
    """
    VARIANT 2: Without Summary Module
    Raw review text fed DIRECTLY to Review Analysis Agent — no chunking, no summarization.
    Reference: same raw review text (what the agent actually sees)
    This tests whether the Summary Module adds value.
    """
    print(f"\n{'='*60}")
    print(f"🔬 VARIANT: Without Summary Module")
    print(f"{'='*60}\n")

    team = TeamRevify()
    df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
    review_dicts = df_filtered.to_dict(orient='records')

    metrics = {
        'variant': 'no_summary_module',
        'total_reviews': len(review_dicts),
        'run_timestamp': datetime.now().isoformat(),
        'summarization_time': 0,  # no summarization
        'analysis_time': 0,
        'total_time': 0,
        'num_chunks': 0,          # no chunks
        'parsing_success': False,
    }

    start_total = time.time()

    # Skip Summary Module entirely — concatenate raw reviews directly
    print(f"⚡ Skipping Summary Module. Concatenating raw reviews directly...")
    raw_reviews_text = "\n\n".join(
        f"Rating: {r.get('reviews.rating', 'N/A')}\n"
        f"Title: {r.get('reviews.title', 'N/A')}\n"
        f"Review: {r.get('reviews.text', 'N/A')}"
        for r in review_dicts
        if r.get('reviews.text')
    )

    # Reference is the raw review text itself (what the agent sees)
    reference_text = raw_reviews_text
    print(f"📄 Raw reviews: {len(raw_reviews_text.split())} words, no summarization")

    # Phase 2: Analysis directly on raw reviews
    print(f"\n🤖 Running Review Analysis Agent on raw reviews...")
    t0 = time.time()

    review_agent  = team.review_analysis_agent()
    analysis_task = team.comprehensive_review_analysis_task()
    analysis_crew = Crew(agents=[review_agent], tasks=[analysis_task],
                         process=Process.sequential, verbose=False)

    result = analysis_crew.kickoff(inputs={
        "features": ", ".join(FEATURES),
        "reviews":  raw_reviews_text
    })

    metrics['analysis_time'] = round(time.time() - t0, 2)
    metrics['total_time']    = round(time.time() - start_total, 2)

    raw_output = result.raw
    json_str   = extract_json_from_markdown(raw_output)

    try:
        analysis_results = json.loads(json_str)
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results] if isinstance(analysis_results, dict) else []
        metrics['parsing_success'] = True
    except Exception as e:
        analysis_results = []
        metrics['parsing_success'] = False

    hypothesis_text = extract_hypothesis(raw_output, analysis_results)

    return metrics, reference_text, hypothesis_text, analysis_results


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=15, max=120),
    stop=stop_after_attempt(5)
)
def run_no_feature_extraction(reviews_df):
    """
    VARIANT 3: Without Feature Extraction Module
    Full chunked summarization pipeline runs as normal, BUT no predefined
    feature list is passed. The Analysis Agent identifies features itself.
    This tests whether pre-extracting features improves analysis quality.
    """
    print(f"\n{'='*60}")
    print(f"🔬 VARIANT: Without Feature Extraction  [chunk_size={CHUNK_SIZE}]")
    print(f"{'='*60}\n")

    team = TeamRevify()
    df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
    review_dicts = df_filtered.to_dict(orient='records')

    metrics = {
        'variant': 'no_feature_extraction',
        'total_reviews': len(review_dicts),
        'run_timestamp': datetime.now().isoformat(),
        'summarization_time': 0,
        'analysis_time': 0,
        'total_time': 0,
        'num_chunks': 0,
        'parsing_success': False,
    }

    start_total = time.time()

    # Phase 1: Chunked summarization runs normally
    print(f"📝 Summarising {len(review_dicts)} reviews [chunk_size={CHUNK_SIZE}]...")
    t0 = time.time()
    chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=CHUNK_SIZE)
    metrics['summarization_time'] = round(time.time() - t0, 2)
    metrics['num_chunks'] = len(chunk_summaries)

    reference_text = "\n\n".join(chunk_summaries)
    reviews_input  = reference_text
    print(f"✅ {len(chunk_summaries)} chunk summaries ({metrics['summarization_time']}s)")

    # Phase 2: Analysis WITHOUT predefined features
    # Agent must identify features autonomously from reviews
    print(f"\n🤖 Running Review Analysis Agent WITHOUT predefined features...")
    t0 = time.time()

    review_agent  = team.review_analysis_agent()
    analysis_task = team.comprehensive_review_analysis_task()
    analysis_crew = Crew(agents=[review_agent], tasks=[analysis_task],
                         process=Process.sequential, verbose=False)

    result = analysis_crew.kickoff(inputs={
        # No predefined features — agent identifies them autonomously
        "features": "Identify the most important features yourself based on what customers frequently mention in the reviews.",
        "reviews":  reviews_input
    })

    metrics['analysis_time'] = round(time.time() - t0, 2)
    metrics['total_time']    = round(time.time() - start_total, 2)

    raw_output = result.raw
    json_str   = extract_json_from_markdown(raw_output)

    try:
        analysis_results = json.loads(json_str)
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results] if isinstance(analysis_results, dict) else []
        metrics['parsing_success'] = True
    except Exception as e:
        analysis_results = []
        metrics['parsing_success'] = False

    hypothesis_text = extract_hypothesis(raw_output, analysis_results)

    return metrics, reference_text, hypothesis_text, analysis_results


def save_and_score(variant_name, metrics, reference_text, hypothesis_text, analysis_results):
    """Compute scores, save JSON, append to CSV."""
    rouge  = compute_rouge_scores(hypothesis=hypothesis_text, reference=reference_text)
    bert   = compute_bert_score(hypothesis=hypothesis_text,   reference=reference_text)
    scores = {**rouge, **bert}

    print(f"\n📈 ROUGE-1: {rouge['rouge1_f1']}  ROUGE-2: {rouge['rouge2_f1']}  "
          f"ROUGE-L: {rouge['rougeL_f1']}  BERT: {bert['bert_f1']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f"{variant_name}_results_{timestamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            'metrics':    metrics,
            'scores':     scores,
            'reference':  reference_text,
            'hypothesis': hypothesis_text,
            'analysis':   analysis_results
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {result_file}")

    append_to_csv({**metrics, **scores})
    return scores


def main():
    print(f"\n{'='*60}")
    print(f"🚀 REVIFY PARTS ABLATION  —  VARIANT: {VARIANT}")
    print(f"{'='*60}\n")

    try:
        reviews_df = pd.read_csv(REVIEWS_FILE)
        print(f"✅ Loaded {len(reviews_df)} reviews from {REVIEWS_FILE}\n")
    except FileNotFoundError:
        print(f"❌ {REVIEWS_FILE} not found.")
        return

    if VARIANT == "full":
        metrics, reference_text, hypothesis_text, analysis_results = run_full_revify(reviews_df)
    elif VARIANT == "no_summary":
        metrics, reference_text, hypothesis_text, analysis_results = run_no_summary_module(reviews_df)
    elif VARIANT == "no_feature_extraction":
        metrics, reference_text, hypothesis_text, analysis_results = run_no_feature_extraction(reviews_df)
    else:
        print(f"❌ Unknown VARIANT '{VARIANT}'. Choose: full | no_summary | no_feature_extraction")
        return

    scores = save_and_score(VARIANT, metrics, reference_text, hypothesis_text, analysis_results)

    print(f"\n{'='*60}")
    print(f"✅ DONE  —  {VARIANT}")
    print(f"   Total time : {metrics['total_time']}s")
    print(f"   ROUGE-1    : {scores['rouge1_f1']}")
    print(f"   ROUGE-2    : {scores['rouge2_f1']}")
    print(f"   ROUGE-L    : {scores['rougeL_f1']}")
    print(f"   BERT F1    : {scores['bert_f1']}")
    print(f"   CSV        : {SCORES_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()