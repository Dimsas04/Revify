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
REVIEWS_FILE   = "benchmark_reviews_250.csv"
CHUNK_SIZE     = 10           # <-- change this per run
OUTPUT_DIR     = "benchmark_output"
SCORES_CSV     = os.path.join(OUTPUT_DIR, "ablation_scores.csv")

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
    """Compute BERTScore F1."""
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {'bert_f1': round(F1.mean().item(), 4)}


def compute_rouge_scores(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1_f1': round(scores['rouge1'].fmeasure, 4),
        'rouge2_f1': round(scores['rouge2'].fmeasure, 4),
        'rougeL_f1': round(scores['rougeL'].fmeasure, 4),
    }


def append_to_csv(row: dict):
    """Append a result row to the scores CSV, creating headers if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(SCORES_CSV)
    df_row.to_csv(SCORES_CSV, mode='a', header=write_header, index=False)
    print(f"📊 Scores appended to: {SCORES_CSV}")


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=15, max=120),
    stop=stop_after_attempt(5)
)
def run_benchmark(chunk_size, reviews_df):
    team = TeamRevify()

    df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
    review_dicts = df_filtered.to_dict(orient='records')

    # Build a simple reference: concatenate all review texts
    reference_text = " ".join(
        str(r.get('reviews.text', '')) for r in review_dicts if r.get('reviews.text')
    )

    metrics = {
        'chunk_size':          chunk_size,
        'total_reviews':       len(review_dicts),
        'run_timestamp':       datetime.now().isoformat(),
        'summarization_time':  0,
        'analysis_time':       0,
        'total_time':          0,
        'num_chunks':          0,
        'tokens_estimate':     0,
        'parsing_success':     False,
    }

    start_total = time.time()

    # ── Phase 1: Summarisation ──────────────────────────────
    print(f"\n📝 Summarising {len(review_dicts)} reviews  [chunk_size={chunk_size}]...")
    t0 = time.time()
    chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=chunk_size)
    metrics['summarization_time'] = round(time.time() - t0, 2)
    metrics['num_chunks']         = len(chunk_summaries)

    reviews_input             = "\n\n".join(chunk_summaries)
    metrics['tokens_estimate'] = len(reviews_input.split())
    print(f"✅ {len(chunk_summaries)} chunk summaries  ({metrics['summarization_time']}s)")

    # ── Phase 2: Final Analysis ─────────────────────────────
    print(f"\n🤖 Running final AI analysis...")
    t0 = time.time()

    review_agent  = team.review_analysis_agent()
    analysis_task = team.comprehensive_review_analysis_task()

    analysis_crew = Crew(
        agents=[review_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=False
    )

    result = analysis_crew.kickoff(inputs={
        "features": ", ".join(FEATURES),
        "reviews":  reviews_input
    })

    metrics['analysis_time'] = round(time.time() - t0, 2)
    metrics['total_time']    = round(time.time() - start_total, 2)
    print(f"⏱️  Analysis: {metrics['analysis_time']}s  |  Total: {metrics['total_time']}s")

    # ── Parse JSON output ───────────────────────────────────
    raw_output = result.raw
    json_str   = extract_json_from_markdown(raw_output)

    try:
        analysis_results = json.loads(json_str)
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results] if isinstance(analysis_results, dict) else []
        metrics['parsing_success'] = True
    except Exception as e:
        print(f"⚠️  JSON parsing failed: {e}")
        analysis_results          = []
        metrics['parsing_error']  = str(e)

    # ── ROUGE scores ────────────────────────────────────────
    # Use the full model output text as hypothesis vs raw reviews as reference
    rouge = compute_rouge_scores(hypothesis=raw_output, reference=reference_text)
    bert  = compute_bert_score(hypothesis=raw_output, reference=reference_text)
    scores = {**rouge, **bert}
    print(f"📈 ROUGE-1 F1: {rouge['rouge1_f1']}  ROUGE-2 F1: {rouge['rouge2_f1']}  ROUGE-L F1: {rouge['rougeL_f1']}  BERT F1: {bert['bert_f1']}")

    # ── Save full results JSON ──────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_size}_results_{timestamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({'metrics': metrics, 'scores': scores, 'analysis': analysis_results},
                  f, indent=2, ensure_ascii=False)
    print(f"💾 Full results: {result_file}")

    # ── Append to running CSV ───────────────────────────────
    csv_row = {**metrics, **scores}
    append_to_csv(csv_row)

    return metrics, scores, analysis_results


def main():
    print(f"\n{'='*60}")
    print(f"🚀 REVIFY ABLATION STUDY  —  chunk_size={CHUNK_SIZE}")
    print(f"{'='*60}\n")

    try:
        reviews_df = pd.read_csv(REVIEWS_FILE)
        print(f"✅ Loaded {len(reviews_df)} reviews from {REVIEWS_FILE}\n")
    except FileNotFoundError:
        print(f"❌ {REVIEWS_FILE} not found. Place it in the project root.")
        return

    metrics, scores, _ = run_benchmark(CHUNK_SIZE, reviews_df)

    print(f"\n{'='*60}")
    print(f"✅ DONE")
    print(f"   Chunk size      : {metrics['chunk_size']}")
    print(f"   Num chunks      : {metrics['num_chunks']}")
    print(f"   Total time      : {metrics['total_time']}s")
    print(f"   ROUGE-1 F1      : {scores['rouge1_f1']}")
    print(f"   ROUGE-2 F1      : {scores['rouge2_f1']}")
    print(f"   ROUGE-L F1      : {scores['rougeL_f1']}")
    print(f"   BERT F1         : {scores['bert_f1']}")
    print(f"   Scores CSV      : {SCORES_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
