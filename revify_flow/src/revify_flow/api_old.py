from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import sys
import threading
import time
from datetime import datetime
import traceback
import re
from crewai import Crew, Process
from .services.analysis_service import (
    create_analysis_request,
    get_or_create_product,
    get_product_reviews_by_asin,
    list_user_analysis_requests,
    update_analysis_request,
)
from .services.supabase_client import get_authenticated_user
from .tools.apifyScrapper_tool import scrape_and_persist_reviews

# Add your project to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import your existing functions
from .main import run_workflow, extract_json_from_markdown, summarize_reviews_chunked
from .crews.team_revify.team_revify import TeamRevify

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your Next.js frontend


def authenticated_user():
    """Return the Supabase user represented by the bearer token, if valid."""
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return None
    return get_authenticated_user(authorization.removeprefix("Bearer ").strip())

# Global variable to track analysis status
analysis_status = {
    'is_running': False,
    'progress': 0,
    'current_phase': '',
    'error': None,
    'result': None,
    'start_time': None,
    'reviews_cached': False,  # Track if reviews are cached for current session
    'cached_product_url': None  # Track which product's reviews are cached
}

def reset_status():
    """Reset the analysis status (preserves review cache info)"""
    global analysis_status
    analysis_status['is_running'] = False
    analysis_status['progress'] = 0
    analysis_status['current_phase'] = ''
    analysis_status['error'] = None
    analysis_status['result'] = None
    analysis_status['start_time'] = None
    # Keep reviews_cached and cached_product_url for session continuity

def update_status(progress, phase, error=None):
    """Update the analysis status"""
    global analysis_status
    analysis_status['progress'] = progress
    analysis_status['current_phase'] = phase
    if error:
        analysis_status['error'] = error
        analysis_status['is_running'] = False

def run_analysis_workflow(product_url, product_name, selected_features=None):
    """Run the analysis workflow in a separate thread"""
    try:
        global analysis_status
        analysis_status['is_running'] = True
        analysis_status['start_time'] = datetime.now()
        
        update_status(10, "Initializing TeamRevify...")
        team = TeamRevify()

        # Check if features are provided
        if selected_features:
            features = selected_features
            update_status(30, f"Using {len(features)} selected features...")
            print(f"✅ Using user-selected features: {features}")
        else: 
            # Phase 1: Extract Features
            update_status(20, "Extracting product features...")
            
            feature_agent = team.feature_extractor()
            feature_task = team.extract_features_task()
            
            feature_crew = Crew(
                agents=[feature_agent],
                tasks=[feature_task],
                process=Process.sequential,
                verbose=False  # Reduce console output
            )
            
            feature_result = feature_crew.kickoff(inputs={"product_input": product_url})
            feature_raw = feature_result.raw
            
            update_status(40, "Processing extracted features...")
            
            # Process features
            extracted_json = extract_json_from_markdown(feature_raw)
            features_data = json.loads(extracted_json)
            
            if isinstance(features_data, dict) and "features" in features_data:
                features = features_data["features"]
            else:
                features = features_data
            
            # Limit features to avoid rate limits
            if len(features) > 7:
                features = features[:7]
        
        update_status(50, f"Scraping reviews for {len(features)} features...")
        
        # Phase 2: Scrape Reviews
        review_scraper = team.review_scraper()
        scrape_reviews_task = team.scrape_reviews_task()
        
        scrape_crew = Crew(
            agents=[review_scraper],
            tasks=[scrape_reviews_task],
            process=Process.sequential,
            verbose=False
        )
        
        scrape_result = scrape_crew.kickoff(inputs={
            "product_url": product_url,
            "target_reviews": 5,  # Reasonable number for demo
            "product_name": product_name or "Product"
        })
        
        update_status(70, "Processing scraped reviews...")
        
        # Phase 3: Load persisted reviews from Supabase.
        df = load_reviews_from_supabase(product_url)
        df_filtered = df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
        review_dicts = df_filtered.to_dict(orient='records')
        
        update_status(80, "Analyzing reviews by features...")
        
        # Phase 4: Analyze reviews
        chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=30)
        reviews_input = "\n\n".join(chunk_summaries)
        
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
        
        update_status(95, "Finalizing results...")
        
        # Process results with better error handling
        raw_output = result.raw
        print(f"DEBUG: Raw output preview: {raw_output[:500]}...")
        
        # Extract and clean JSON
        json_str = extract_json_from_markdown(raw_output)
        print(f"DEBUG: Extracted JSON preview: {json_str[:500]}...")
        
        # Try to parse JSON with multiple fallback strategies
        analysis_results = None


        # Strategy 1: Direct parsing
        try:
            analysis_results = json.loads(json_str)
            print("✅ JSON parsed successfully on first attempt")
        except json.JSONDecodeError as e:
            print(f"⚠️ First JSON parse attempt failed: {e}")
            
            # Strategy 2: Try with additional cleaning
            try:
                # More aggressive cleaning
                cleaned_json = json_str.replace('\\"', '"')  # Fix over-escaped quotes
                cleaned_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_json)  # Remove control chars
                analysis_results = json.loads(cleaned_json)
                print("✅ JSON parsed successfully after cleaning")
            except json.JSONDecodeError as e2:
                print(f"⚠️ Second JSON parse attempt failed: {e2}")
                
                # Strategy 3: Try to extract just the array part if it's wrapped
                try:
                    # Look for array pattern
                    array_match = re.search(r'\[(.*)\]', json_str, re.DOTALL)
                    if array_match:
                        array_content = '[' + array_match.group(1) + ']'
                        analysis_results = json.loads(array_content)
                        print("✅ JSON parsed successfully by extracting array")
                    else:
                        raise Exception("No valid JSON array found")
                except Exception as e3:
                    print(f"⚠️ Third JSON parse attempt failed: {e3}")
                    
                    # Strategy 4: Create a fallback result
                    print("Creating fallback result from raw text...")
                    analysis_results = create_fallback_result(raw_output, features)
        
        if not analysis_results:
            raise Exception("Could not parse analysis results into valid JSON")
        
        # Ensure analysis_results is a list
        if not isinstance(analysis_results, list):
            if isinstance(analysis_results, dict):
                analysis_results = [analysis_results]
            else:
                raise Exception("Analysis results is not in expected format")

        # Save results
        os.makedirs("output", exist_ok=True)
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/feature_analysis_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        update_status(100, "Analysis completed successfully!")
        
        # Set the result
        analysis_status['result'] = {
            'features': features,
            'analysis': analysis_results,
            'total_reviews': len(review_dicts),
            'filename': filename
        }
        analysis_status['is_running'] = False
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(f"Analysis error: {error_msg}")
        print(traceback.format_exc())
        update_status(0, "Analysis failed", error_msg)

import threading
from queue import Queue
import pandas as pd


def _product_asin(product_url):
    match = re.search(
        r"(?:/dp/|/gp/product/|/product-reviews/)([A-Z0-9]{10})",
        product_url,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def _reviews_dataframe(product, reviews):
    if not product or not reviews:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "name": product.get("title") or "Amazon Product",
            "brand": product.get("brand") or "Unknown",
            "reviews.rating": review.get("rating"),
            "reviews.title": review.get("title") or "",
            "reviews.text": review.get("content") or "",
        }
        for review in reviews
        if review.get("content")
    ])


def load_reviews_from_supabase(product_url, max_reviews):
    asin = _product_asin(product_url)
    if not asin:
        raise ValueError("The product URL does not contain a supported Amazon ASIN")
    product, reviews = get_product_reviews_by_asin(asin)
    if not reviews:
        scraped = scrape_and_persist_reviews(product_url, max_reviews)
        product = scraped["product"]
        reviews = scraped["reviews"]
    reviews_df = _reviews_dataframe(product, reviews)
    if reviews_df.empty:
        raise ValueError("No review content is available for this product")
    return reviews_df


def run_analysis_workflow_optimized(product_url, product_name, selected_features=None,
                                    user_id=None, analysis_request_id=None):
    """Optimized parallel workflow"""
    try:
        global analysis_status
        analysis_status['is_running'] = True
        analysis_status['start_time'] = datetime.now()
        
        update_status(10, "Initializing parallel tasks...")
        team = TeamRevify()
        
        result_queue = Queue()
        features = None
        reviews_df = None

        if selected_features:
                # User already selected features - use pre-scraped reviews
                features = selected_features
                update_status(20, "Using pre-selected features...")
                
                # Check if reviews are cached for THIS specific product
                if (analysis_status['reviews_cached'] and
                    analysis_status['cached_product_url'] == product_url):
                    update_status(30, "Loading pre-scraped reviews...")
                    reviews_df = load_reviews_from_supabase(product_url)
                    print(f"✅ Using pre-scraped reviews: {len(reviews_df)} reviews loaded")
                else:
                    update_status(30, "Loading reviews from Supabase...")
                    scrape_thread = threading.Thread(
                        target=scrape_reviews_parallel,
                        args=(product_url, product_name, result_queue, update_status)
                    )
                    scrape_thread.start()
                    scrape_thread.join()
                    
                    # Collect scraping results
                    while not result_queue.empty():
                        result_type, result_data = result_queue.get()
                        if result_type == "reviews":
                            reviews_df = result_data
                    
                    # Update cache markers
                    analysis_status['reviews_cached'] = True
                    analysis_status['cached_product_url'] = product_url
        else:
                # Run feature extraction AND review scraping in PARALLEL
                update_status(20, "Running feature extraction and review scraping in parallel...")
                
                feature_thread = threading.Thread(
                    target=extract_features_parallel,
                    args=(product_url, result_queue, update_status)
                )
                
                scrape_thread = threading.Thread(
                    target=scrape_reviews_parallel,
                    args=(product_url, product_name, result_queue, update_status)
                )
                
                # Start both threads
                feature_thread.start()
                scrape_thread.start()
                
                # Wait for both to complete
                feature_thread.join()
                scrape_thread.join()

                # Collect results from queue
                while not result_queue.empty():
                    result_type, result_data = result_queue.get()
                    if result_type == "features":
                        features = result_data
                    elif result_type == "reviews":
                        reviews_df = result_data
        
        if features is None:
                raise Exception("Feature extraction failed")
        if reviews_df is None or reviews_df.empty:
            raise Exception("No reviews were scraped")
        
        if len(features) > 10:
                features = features[:10]
            
        update_status(70, f"Processing {len(reviews_df)} reviews for {len(features)} features...")

        # ══════════════════════════════════════════════════
        # PHASE 2: INCREMENTAL SUMMARIZATION
        # ══════════════════════════════════════════════════
        
        # Process reviews in chunks (already done during scraping)
        df_filtered = reviews_df[['name', 'brand', 'reviews.rating', 'reviews.title', 'reviews.text']]
        review_dicts = df_filtered.to_dict(orient='records')
        
        # Summarize in batches
        chunk_summaries = summarize_reviews_chunked(review_dicts, team, chunk_size=30)
        reviews_input = "\n\n".join(chunk_summaries)

        # ══════════════════════════════════════════════════
        # PHASE 3: FINAL ANALYSIS
        # ══════════════════════════════════════════════════
        
        update_status(80, "Running final AI analysis...")
        
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
        
        # Process results with better error handling
        raw_output = result.raw
        print(f"DEBUG: Raw output preview: {raw_output[:500]}...")
        
        # Extract and clean JSON
        json_str = extract_json_from_markdown(raw_output)
        print(f"DEBUG: Extracted JSON preview: {json_str[:500]}...")
        
        # Try to parse JSON with multiple fallback strategies
        analysis_results = None

        # Strategy 1: Direct parsing
        try:
            analysis_results = json.loads(json_str)
            print("✅ JSON parsed successfully on first attempt")
        except json.JSONDecodeError as e:
            print(f"⚠️ First JSON parse attempt failed: {e}")
            
            # Strategy 2: Try with additional cleaning
            try:
                # More aggressive cleaning
                cleaned_json = json_str.replace('\\"', '"')  # Fix over-escaped quotes
                cleaned_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_json)  # Remove control chars
                analysis_results = json.loads(cleaned_json)
                print("✅ JSON parsed successfully after cleaning")
            except json.JSONDecodeError as e2:
                print(f"⚠️ Second JSON parse attempt failed: {e2}")
                
                # Strategy 3: Try to extract just the array part if it's wrapped
                try:
                    # Look for array pattern
                    array_match = re.search(r'\[(.*)\]', json_str, re.DOTALL)
                    if array_match:
                        array_content = '[' + array_match.group(1) + ']'
                        analysis_results = json.loads(array_content)
                        print("✅ JSON parsed successfully by extracting array")
                    else:
                        raise Exception("No valid JSON array found")
                except Exception as e3:
                    print(f"⚠️ Third JSON parse attempt failed: {e3}")
                    
                    # Strategy 4: Create a fallback result
                    print("Creating fallback result from raw text...")
                    analysis_results = create_fallback_result(raw_output, features)
        
        if not analysis_results:
            raise Exception("Could not parse analysis results into valid JSON")
        
        # Ensure analysis_results is a list
        if not isinstance(analysis_results, list):
            if isinstance(analysis_results, dict):
                analysis_results = [analysis_results]
            else:
                raise Exception("Analysis results is not in expected format")

        # Save results
        os.makedirs("output", exist_ok=True)
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/feature_analysis_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        update_status(100, "Analysis completed successfully!")
        
        # Set the result
        analysis_status['result'] = {
            'features': features,
            'analysis': analysis_results,
            'total_reviews': len(review_dicts),
            'filename': filename
        }
        analysis_status['is_running'] = False
        if analysis_request_id:
            update_analysis_request(analysis_request_id, status="completed",
                                    result=analysis_status['result'])

    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(f"Analysis error: {error_msg}")
        print(traceback.format_exc())
        update_status(0, "Analysis failed", error_msg)
        if analysis_request_id:
            try:
                update_analysis_request(analysis_request_id, status="failed", error=error_msg)
            except Exception as persistence_error:
                print(f"Failed to persist analysis error: {persistence_error}")

def extract_features_parallel(product_url, result_queue, update_callback):
    """Extract features in parallel thread"""
    try:
        update_callback(30, "🔍 Extracting features...")
        
        team = TeamRevify()
        feature_agent = team.feature_extractor()
        feature_task = team.extract_features_task()
        
        feature_crew = Crew(
            agents=[feature_agent],
            tasks=[feature_task],
            process=Process.sequential,
            verbose=False
        )
        
        feature_result = feature_crew.kickoff(inputs={"product_input": product_url})
        
        # Process features
        extracted_json = extract_json_from_markdown(feature_result.raw)
        features_data = json.loads(extracted_json)
        
        if isinstance(features_data, dict) and "features" in features_data:
            features = features_data["features"]
        else:
            features = features_data
        
        result_queue.put(("features", features))
        print(f"✅ Feature extraction complete: {len(features)} features found")
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        result_queue.put(("features", None))

def scrape_reviews_parallel(product_url, product_name, result_queue, update_callback):
    """Scrape reviews in parallel thread with incremental processing"""
    try:
        update_callback(40, "📚 Scraping reviews...")
        
        team = TeamRevify()
        review_scraper = team.review_scraper()
        scrape_reviews_task = team.scrape_reviews_task()
        
        scrape_crew = Crew(
            agents=[review_scraper],
            tasks=[scrape_reviews_task],
            process=Process.sequential,
            verbose=False
        )
        
        # Scrape reviews
        scrape_result = scrape_crew.kickoff(inputs={
            "product_url": product_url,
            "target_reviews": 50,
            "product_name": product_name or "Product"
        })
        
        reviews_df = load_reviews_from_supabase(product_url)
        result_queue.put(("reviews", reviews_df))
        print(f"✅ Reviews loaded from Supabase: {len(reviews_df)} reviews")
        
    except Exception as e:
        print(f"❌ Review scraping error: {e}")
        result_queue.put(("reviews", None))

        

def create_fallback_result(raw_output, features):
    """Create a fallback result when JSON parsing fails"""
    fallback_results = []
    
    # Try to extract information from raw text
    lines = raw_output.split('\n')
    current_feature = None
    current_analysis = {}
    
    for line in lines:
        line = line.strip()
        
        # Look for feature names
        for feature in features:
            if feature.lower() in line.lower():
                if current_feature and current_analysis:
                    fallback_results.append(current_analysis)
                
                current_feature = feature
                current_analysis = {
                    "feature": feature,
                    "sentiment": "Mixed",
                    "verdict": "Analysis could not be fully parsed due to formatting issues.",
                    "key_points": ["Raw analysis data available in logs"]
                }
                break
    
    # Add the last analysis if exists
    if current_feature and current_analysis:
        fallback_results.append(current_analysis)
    
    # If no features found, create generic fallback
    if not fallback_results:
        for feature in features:
            fallback_results.append({
                "feature": feature,
                "sentiment": "Unknown",
                "verdict": "Analysis could not be completed due to data formatting issues.",
                "key_points": ["Please try again or check the raw output logs"]
            })
    
    return fallback_results


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Revify API is running"})

# Global storage for feature extraction results
feature_extraction_status = {
    'is_running': False,
    'completed': False,
    'features': None,
    'error': None,
    'reviews_ready': False
}

def run_feature_extraction_async(product_url, product_name):
    """Run feature extraction and review scraping in background"""
    global feature_extraction_status
    
    try:
        feature_extraction_status['is_running'] = True
        feature_extraction_status['completed'] = False
        feature_extraction_status['features'] = None
        feature_extraction_status['error'] = None
        feature_extraction_status['reviews_ready'] = False
        
        update_status(20, "Starting parallel feature extraction and review scraping...")
        
        # Use Queue for thread communication
        result_queue = Queue()
        
        # Create threads for parallel execution
        feature_thread = threading.Thread(
            target=extract_features_parallel,
            args=(product_url, result_queue, update_status)
        )
        
        scrape_thread = threading.Thread(
            target=scrape_reviews_parallel,
            args=(product_url, product_name, result_queue, update_status)
        )
        
        # Start both threads simultaneously
        feature_thread.start()
        scrape_thread.start()
        
        # Poll for features to become available first (don't wait for scraping)
        features = None
        reviews_df = None
        features_found = False
        
        # Check queue periodically for features
        while feature_thread.is_alive() or not result_queue.empty():
            if not result_queue.empty():
                result_type, result_data = result_queue.get()
                if result_type == "features" and result_data:
                    features = result_data
                    features_found = True
                    # Store features immediately so frontend can access them
                    feature_extraction_status['features'] = features
                    feature_extraction_status['completed'] = True
                    print(f"✅ Features available for selection while scraping continues...")
                    break
            time.sleep(0.5)  # Check every 500ms
        
        # Now wait for scraping to complete in background
        scrape_thread.join()
        
        # Collect scraping results
        while not result_queue.empty():
            result_type, result_data = result_queue.get()
            if result_type == "reviews":
                reviews_df = result_data
        
        if features is None:
            feature_extraction_status['error'] = "Feature extraction failed"
            feature_extraction_status['is_running'] = False
            return
        
        # Store results
        feature_extraction_status['features'] = features
        feature_extraction_status['completed'] = True
        feature_extraction_status['is_running'] = False
        
        # Store reviews_df globally for later use
        if reviews_df is not None and not reviews_df.empty:
            feature_extraction_status['reviews_ready'] = True
            
            # Mark reviews as cached for this session
            analysis_status['reviews_cached'] = True
            analysis_status['cached_product_url'] = product_url
            
            print(f"✅ {len(reviews_df)} reviews available from Supabase")
        
    except Exception as e:
        feature_extraction_status['error'] = str(e)
        feature_extraction_status['is_running'] = False
        feature_extraction_status['completed'] = False

@app.route('/api/extract-features', methods=['POST'])
def extract_features_only():
    """Start feature extraction AND review scraping in parallel (async)"""
    try:
        if not authenticated_user():
            return jsonify({"error": "Authentication required"}), 401
        global feature_extraction_status
        
        data = request.get_json()
        
        if not data or 'product_url' not in data:
            return jsonify({"error": "product_url is required"}), 400
        
        product_url = data['product_url']
        product_name = data.get('product_name', 'Product')
        
        # Check if extraction is already running
        if feature_extraction_status['is_running']:
            return jsonify({"error": "Feature extraction already in progress"}), 409
        
        # Clear cache if analyzing a different product
        if analysis_status['cached_product_url'] != product_url:
            analysis_status['reviews_cached'] = False
            analysis_status['cached_product_url'] = None
            print(f"🔄 Starting fresh analysis for new product")
        
        # Start extraction in background thread
        extraction_thread = threading.Thread(
            target=run_feature_extraction_async,
            args=(product_url, product_name)
        )
        extraction_thread.daemon = True
        extraction_thread.start()
        
        return jsonify({
            "message": "Feature extraction started",
            "status": "running"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to start feature extraction: {str(e)}"}), 500

@app.route('/api/feature-status', methods=['GET'])
def get_feature_extraction_status():
    """Get feature extraction status"""
    if not authenticated_user():
        return jsonify({"error": "Authentication required"}), 401
    global feature_extraction_status
    
    response = {
        'is_running': feature_extraction_status['is_running'],
        'completed': feature_extraction_status['completed'],
        'error': feature_extraction_status['error'],
        'reviews_ready': feature_extraction_status['reviews_ready']
    }
    
    if feature_extraction_status['completed'] and feature_extraction_status['features']:
        response['features'] = feature_extraction_status['features']
        response['total_count'] = len(feature_extraction_status['features'])
    
    return jsonify(response)


@app.route('/api/analyze', methods=['POST'])
def analyze_product():
    """Start product analysis"""
    try:
        user = authenticated_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        data = request.get_json()
        
        if not data or 'product_url' not in data:
            return jsonify({"error": "product_url is required"}), 400
        
        product_url = data['product_url']
        product_name = data.get('product_name', '')
        selected_features = data.get('selected_features', None)  # NEW: Accept selected features
        
        # Check if analysis is already running
        if analysis_status['is_running']:
            return jsonify({"error": "Analysis is already running"}), 409
        
        # Reset status and start analysis in background thread
        reset_status()
        asin = _product_asin(product_url)
        product_id = None
        if asin:
            product = get_or_create_product(
                asin,
                product_url,
                product_name or None,
            )
            product_id = product["id"]
        else:
            return jsonify({"error": "The product URL does not contain a supported Amazon ASIN"}), 400
        analysis_request_id = create_analysis_request(
            str(user.id),
            product_url,
            product_name,
            selected_features,
            product_id,
        )
        
        # Start analysis in a separate thread
        analysis_thread = threading.Thread(
            target=run_analysis_workflow_optimized, 
            args=(product_url, product_name, selected_features, str(user.id), analysis_request_id)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            "message": "Analysis started successfully",
            "status": "running",
            "features_count": len(selected_features) if selected_features else "all"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to start analysis: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_analysis_status():
    """Get current analysis status"""
    if not authenticated_user():
        return jsonify({"error": "Authentication required"}), 401
    return jsonify(analysis_status)

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get analysis results"""
    if not authenticated_user():
        return jsonify({"error": "Authentication required"}), 401
    if analysis_status['result']:
        return jsonify(analysis_status['result'])
    else:
        return jsonify({"error": "No results available"}), 404


@app.route('/api/history', methods=['GET'])
def get_analysis_history():
    user = authenticated_user()
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    try:
        return jsonify(list_user_analysis_requests(str(user.id)))
    except Exception as error:
        return jsonify({"error": f"Failed to retrieve analysis history: {error}"}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download analysis file"""
    try:
        file_path = os.path.join("output", filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)