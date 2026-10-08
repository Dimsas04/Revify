from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import traceback

from .services.supabase_client import get_authenticated_user
from .services.analysis_service import (
    create_analysis_request,
    get_analysis_request,
    get_or_create_product,
    list_user_analysis_requests,
    update_analysis_request,
    update_analysis_request_features,
)

from .utils.amazon import extract_asin

from .workflows.revify_preparation_flow import RevifyPreparationFlow
from .workflows.revify_analysis_flow import RevifyAnalysisFlow


app = Flask(__name__)
CORS(app)


# ============================================================
# Authentication
# ============================================================

def authenticated_user():
    """
    Return the Supabase user represented by the bearer token.
    Returns None when authentication fails or the token is missing.
    """
    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        return None

    token = authorization.removeprefix("Bearer ").strip()

    if not token:
        return None

    return get_authenticated_user(token)


# ============================================================
# Background workflow runners
# ============================================================

def run_preparation_workflow(
    analysis_request_id: str,
    product_url: str,
    product_name: str,
):
    """
    Run feature extraction + review acquisition in the background.

    The parallel execution between feature extraction and Zebu
    is handled inside RevifyPreparationFlow.
    """
    try:
        flow = RevifyPreparationFlow(
            analysis_request_id=analysis_request_id,
            product_url=product_url,
            product_name=product_name,
        )

        flow.kickoff()

    except Exception as exc:
        error_message = f"Preparation workflow failed: {exc}"

        print(error_message)
        print(traceback.format_exc())

        try:
            update_analysis_request(
                analysis_request_id,
                status="failed",
                error=error_message,
            )
        except Exception as persistence_error:
            print(
                f"Failed to persist preparation error: "
                f"{persistence_error}"
            )


def run_analysis_workflow(
    analysis_request_id: str,
):
    """
    Run the final review-analysis workflow in the background.

    At this point the product's reviews already exist in Supabase.
    No scraping happens here.
    """
    try:
        flow = RevifyAnalysisFlow(
            analysis_request_id=analysis_request_id,
        )

        flow.kickoff()

    except Exception as exc:
        error_message = f"Analysis workflow failed: {exc}"

        print(error_message)
        print(traceback.format_exc())

        try:
            update_analysis_request(
                analysis_request_id,
                status="failed",
                error=error_message,
            )
        except Exception as persistence_error:
            print(
                f"Failed to persist analysis error: "
                f"{persistence_error}"
            )


# ============================================================
# Health
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Revify API is running",
    })


# ============================================================
# Phase 1
# Feature extraction + review acquisition
# ============================================================

@app.route("/api/extract-features", methods=["POST"])
def extract_features():
    """
    Start the preparation phase.

    This phase runs:
        Feature Extraction   ─┐
                              ├── in parallel
        Zebu Review Fetch     ─┘

    The frontend receives an analysis_request_id and polls
    /api/feature-status.
    """

    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    data = request.get_json() or {}

    product_url = data.get("product_url")
    product_name = data.get("product_name", "")

    if not product_url:
        return jsonify({
            "error": "product_url is required"
        }), 400

    asin = extract_asin(product_url)

    if not asin:
        return jsonify({
            "error": "The product URL does not contain a supported Amazon ASIN"
        }), 400

    try:
        # ----------------------------------------------------
        # Create/retrieve product
        # ----------------------------------------------------

        product = get_or_create_product(
            asin=asin,
            url=product_url,
            title=product_name or None,
        )

        # ----------------------------------------------------
        # Create analysis request
        # ----------------------------------------------------

        analysis_request_id = create_analysis_request(
            user_id=str(user.id),
            product_url=product_url,
            product_name=product_name,
            selected_features=None,
            product_id=product["id"],
        )

        # ----------------------------------------------------
        # Start preparation workflow in background
        # ----------------------------------------------------

        preparation_thread = threading.Thread(
            target=run_preparation_workflow,
            args=(
                analysis_request_id,
                product_url,
                product_name,
            ),
            daemon=True,
        )

        preparation_thread.start()

        return jsonify({
            "message": "Feature extraction and review acquisition started",
            "status": "running",
            "analysis_request_id": analysis_request_id,
        }), 202

    except Exception as exc:
        return jsonify({
            "error": f"Failed to start preparation: {exc}"
        }), 500


# ============================================================
# Phase 1 status
# ============================================================

@app.route("/api/feature-status", methods=["GET"])
def feature_status():
    """
    Return preparation status and extracted features for
    a specific analysis request.
    """

    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    analysis_request_id = request.args.get("analysis_request_id")

    if not analysis_request_id:
        return jsonify({
            "error": "analysis_request_id is required"
        }), 400

    try:
        analysis_request = get_analysis_request(
            analysis_request_id,
            str(user.id),
        )

        if not analysis_request:
            return jsonify({
                "error": "Analysis request not found"
            }), 404

        return jsonify({
            "analysis_request_id": analysis_request["id"],
            "status": analysis_request["status"],
            "progress": analysis_request.get("progress", 0),
            "current_phase": analysis_request.get("current_phase"),
            "features": analysis_request.get("extracted_features"),
            "reviews_ready": analysis_request["status"] == "awaiting_selection",
        })

    except Exception as exc:
        return jsonify({
            "error": f"Failed to retrieve feature status: {exc}"
        }), 500


# ============================================================
# Phase 2
# User selects features → final analysis
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze_product():
    """
    Start the final analysis using the already-acquired reviews
    and the features selected by the user.
    """

    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    data = request.get_json() or {}

    analysis_request_id = data.get("analysis_request_id")
    selected_features = data.get("selected_features")

    if not analysis_request_id:
        return jsonify({
            "error": "analysis_request_id is required"
        }), 400

    if not selected_features or not isinstance(selected_features, list):
        return jsonify({
            "error": "selected_features must be a non-empty array"
        }), 400

    try:
        # ----------------------------------------------------
        # Verify request belongs to authenticated user
        # ----------------------------------------------------

        analysis_request = get_analysis_request(
            analysis_request_id,
            str(user.id),
        )

        if not analysis_request:
            return jsonify({
                "error": "Analysis request not found"
            }), 404

        # ----------------------------------------------------
        # Make sure preparation is complete
        # ----------------------------------------------------

        if analysis_request["status"] != "awaiting_selection":
            return jsonify({
                "error": (
                    "This analysis request is not ready for "
                    "feature selection"
                ),
                "current_status": analysis_request["status"],
            }), 409

        # ----------------------------------------------------
        # Save selected features
        # ----------------------------------------------------

        update_analysis_request_features(
            analysis_request_id,
            selected_features,
        )

        # ----------------------------------------------------
        # Change status
        # ----------------------------------------------------

        update_analysis_request(
            analysis_request_id,
            status="running",
        )

        # ----------------------------------------------------
        # Start final analysis in background
        # ----------------------------------------------------

        analysis_thread = threading.Thread(
            target=run_analysis_workflow,
            args=(analysis_request_id,),
            daemon=True,
        )

        analysis_thread.start()

        return jsonify({
            "message": "Product analysis started",
            "status": "running",
            "analysis_request_id": analysis_request_id,
            "features_count": len(selected_features),
        }), 202

    except Exception as exc:
        return jsonify({
            "error": f"Failed to start analysis: {exc}"
        }), 500


# ============================================================
# Analysis status
# ============================================================

@app.route("/api/status", methods=["GET"])
def get_analysis_status():
    """
    Return the status of a specific analysis request.
    """

    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    analysis_request_id = request.args.get("analysis_request_id")

    if not analysis_request_id:
        return jsonify({
            "error": "analysis_request_id is required"
        }), 400

    try:
        analysis_request = get_analysis_request(
            analysis_request_id,
            str(user.id),
        )

        if not analysis_request:
            return jsonify({
                "error": "Analysis request not found"
            }), 404

        return jsonify({
            "analysis_request_id": analysis_request["id"],
            "status": analysis_request["status"],
            "is_running": analysis_request["status"] == "running",
            "progress": analysis_request.get("progress", 0),
            "current_phase": analysis_request.get("current_phase"),
            "error": analysis_request.get("error"),
            "result": analysis_request.get("result"),
        })

    except Exception as exc:
        return jsonify({
            "error": f"Failed to retrieve analysis status: {exc}"
        }), 500


# ============================================================
# Results
# ============================================================

@app.route("/api/results", methods=["GET"])
def get_results():
    """
    Return completed analysis results from Supabase.
    """

    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    analysis_request_id = request.args.get("analysis_request_id")

    if not analysis_request_id:
        return jsonify({
            "error": "analysis_request_id is required"
        }), 400

    try:
        analysis_request = get_analysis_request(
            analysis_request_id,
            str(user.id),
        )

        if not analysis_request:
            return jsonify({
                "error": "Analysis request not found"
            }), 404

        if analysis_request["status"] != "completed":
            return jsonify({
                "error": "Analysis is not completed yet",
                "status": analysis_request["status"],
            }), 409

        return jsonify({
            "analysis_request_id": analysis_request["id"],
            "result": analysis_request.get("result"),
        })

    except Exception as exc:
        return jsonify({
            "error": f"Failed to retrieve results: {exc}"
        }), 500


# ============================================================
# History
# ============================================================

@app.route("/api/history", methods=["GET"])
def get_analysis_history():
    user = authenticated_user()

    if not user:
        return jsonify({
            "error": "Authentication required"
        }), 401

    try:
        history = list_user_analysis_requests(
            str(user.id)
        )

        return jsonify(history)

    except Exception as exc:
        return jsonify({
            "error": f"Failed to retrieve analysis history: {exc}"
        }), 500


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
    )