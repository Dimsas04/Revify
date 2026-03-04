import pandas as pd
import json
import time
from datetime import datetime
from main import extract_json_from_markdown, summarize_reviews_chunked
from src.revify_flow.crews.team_revify.team_revify import TeamRevify
from crewai import Crew, Process
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm.exceptions import RateLimitError

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=15, max=120),
    stop=stop_after_attempt(5)
)

def run_benchmark_with_retry(chunk_size, reviews_df, features, output_dir="benchmark_output"):
    return run_benchmark(chunk_size, reviews_df, features, output_dir)

def run_benchmark(chunk_size, reviews_df, features, output_dir="benchmark_output"):
    """Run analysis with specific chunk size"""
    
    print(f"\n{'='*60}")
    print(f"🔬 BENCHMARKING: Chunk Size = {chunk_size}")
    print(f"{'='*60}\n")
    
    team = TeamRevify()
    
    # Filter reviews
    df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
    review_dicts = df_filtered.to_dict(orient='records')
    
    # Track metrics
    metrics = {
        'chunk_size': chunk_size,
        'total_reviews': len(review_dicts),
        'start_time': datetime.now().isoformat(),
        'summarization_time': 0,
        'analysis_time': 0,
        'total_time': 0,
        'num_chunks': 0,
        'tokens_estimate': 0
    }
    
    start_total = time.time()
    
    # PHASE 1: Summarization
    print(f"📝 Summarizing {len(review_dicts)} reviews with chunk_size={chunk_size}...")
    start_summarization = time.time()
    
    chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=chunk_size)
    
    metrics['summarization_time'] = time.time() - start_summarization
    metrics['num_chunks'] = len(chunk_summaries)
    
    reviews_input = "\n\n".join(chunk_summaries)
    metrics['tokens_estimate'] = len(reviews_input.split())
    
    print(f"✅ Created {len(chunk_summaries)} chunk summaries")
    print(f"⏱️ Summarization took: {metrics['summarization_time']:.2f}s")
    
    # PHASE 2: Final Analysis
    print(f"\n🤖 Running final AI analysis...")
    start_analysis = time.time()
    
    review_agent = team.review_analysis_agent()
    analysis_task = team.comprehensive_review_analysis_task()
    
    analysis_crew = Crew(
        agents=[review_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=False
    )
    
    result = analysis_crew.kickoff(inputs={
        "features": ", ".join(features),
        "reviews": reviews_input
    })
    
    metrics['analysis_time'] = time.time() - start_analysis
    metrics['total_time'] = time.time() - start_total
    metrics['end_time'] = datetime.now().isoformat()
    
    print(f"⏱️ Analysis took: {metrics['analysis_time']:.2f}s")
    print(f"⏱️ Total time: {metrics['total_time']:.2f}s")
    
    # Parse results
    raw_output = result.raw
    json_str = extract_json_from_markdown(raw_output)
    
    try:
        analysis_results = json.loads(json_str)
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results] if isinstance(analysis_results, dict) else []
        metrics['parsing_success'] = True
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}")
        analysis_results = []
        metrics['parsing_success'] = False
        metrics['parsing_error'] = str(e)
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete analysis results (metrics + feature insights)
    result_file = f"{output_dir}/chunk_{chunk_size}_complete_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            'metrics': metrics,
            'analysis': analysis_results
        }, f, indent=2, ensure_ascii=False)
    
    # Save feature-based insights separately for easier comparison
    insights_file = f"{output_dir}/chunk_{chunk_size}_feature_insights_{timestamp}.json"
    with open(insights_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Complete results saved to: {result_file}")
    print(f"💾 Feature insights saved to: {insights_file}")
    
    return metrics, analysis_results


def main():
    """Run benchmarking with different chunk sizes"""
    
    # Configuration
    REVIEWS_FILE = "benchmark_reviews_250.csv"  # Your 500-review dataset
    CHUNK_SIZES = [25, 50, 125]
    
    # Sample features (you can modify these)
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
    
    print(f"\n{'='*60}")
    print(f"🚀 REVIFY CHUNK SIZE BENCHMARKING")
    print(f"{'='*60}")
    print(f"📊 Reviews: {REVIEWS_FILE}")
    print(f"📏 Chunk sizes: {CHUNK_SIZES}")
    print(f"🎯 Features: {len(FEATURES)}")
    print(f"{'='*60}\n")
    
    # Load reviews
    try:
        reviews_df = pd.read_csv(REVIEWS_FILE)
        print(f"✅ Loaded {len(reviews_df)} reviews from {REVIEWS_FILE}\n")
    except FileNotFoundError:
        print(f"❌ Error: {REVIEWS_FILE} not found!")
        print(f"Please place your 500-review dataset in the revify_flow/ directory")
        return
    
    # Run benchmarks
    all_metrics = []
    all_insights_files = []  # Track insight files for summary
    
    for chunk_size in CHUNK_SIZES:
        try:
            metrics, results = run_benchmark(chunk_size, reviews_df, FEATURES)
            all_metrics.append(metrics)
            
            # Track the insights file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            insights_file = f"benchmark_output/chunk_{chunk_size}_feature_insights_{timestamp}.json"
            all_insights_files.append({
                'chunk_size': chunk_size,
                'file': insights_file,
                'num_features': len(results) if results else 0
            })
            
            # Small delay between runs
            time.sleep(60)
            
        except Exception as e:
            print(f"❌ Benchmark failed for chunk_size={chunk_size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK COMPARISON REPORT")
    print(f"{'='*60}\n")
    
    comparison_df = pd.DataFrame(all_metrics)
    
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_file = f"benchmark_output/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison saved to: {comparison_file}")
    
    # Save insights file summary
    insights_summary_file = f"benchmark_output/insights_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(insights_summary_file, "w", encoding="utf-8") as f:
        json.dump({
            'benchmark_info': {
                'reviews_file': REVIEWS_FILE,
                'chunk_sizes': CHUNK_SIZES,
                'features': FEATURES,
                'total_reviews': len(reviews_df),
                'benchmark_date': datetime.now().isoformat()
            },
            'insight_files': all_insights_files
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Insights summary saved to: {insights_summary_file}")
    
    # Print insights
    print(f"\n{'='*60}")
    print(f"💡 KEY INSIGHTS")
    print(f"{'='*60}")
    
    if len(all_metrics) > 0:
        fastest = min(all_metrics, key=lambda x: x['total_time'])
        print(f"⚡ Fastest: Chunk size {fastest['chunk_size']} ({fastest['total_time']:.2f}s)")
        
        least_chunks = min(all_metrics, key=lambda x: x['num_chunks'])
        print(f"📦 Least chunks: Chunk size {least_chunks['chunk_size']} ({least_chunks['num_chunks']} chunks)")
        
        print(f"\n📁 Feature Insights Files Generated:")
        for insight_info in all_insights_files:
            print(f"   • Chunk {insight_info['chunk_size']}: {insight_info['file']} ({insight_info['num_features']} features)")


if __name__ == "__main__":
    main()