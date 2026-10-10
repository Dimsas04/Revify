from crewai import Crew, Process
from crewai.flow.flow import Flow, start

from ..crews.team_revify.team_revify import TeamRevify

from ..services.analysis_service import (
    get_analysis_request,
    get_product_reviews,
    update_analysis_request,
    update_analysis_progress,
)
from ..services.review_summarization_service import summarize_reviews_chunked


class RevifyAnalysisFlow(Flow):
    """
    Final analysis workflow.

    Preparation has already happened before this Flow starts:
        1. Product exists in Supabase
        2. Reviews have been acquired from Zebu
        3. Reviews have been persisted to Supabase
        4. User has selected the features to analyze

    This Flow:
        1. Loads the analysis request
        2. Loads reviews from Supabase
        3. Summarizes reviews in chunks
        4. Runs the final feature-based analysis
        5. Persists the result
    """

    def __init__(self, analysis_request_id: str):
        super().__init__()

        self.analysis_request_id = analysis_request_id

    @start()
    def run_analysis(self):
        # ---------------------------------------------------------
        # 1. Load analysis request
        # ---------------------------------------------------------

        analysis_request = get_analysis_request(
            self.analysis_request_id
        )

        if not analysis_request:
            raise ValueError(
                f"Analysis request not found: "
                f"{self.analysis_request_id}"
            )

        product_id = analysis_request.get("product_id")
        selected_features = (
            analysis_request.get("selected_features") or []
        )

        if not product_id:
            raise ValueError(
                "Analysis request does not have a product_id"
            )

        if not selected_features:
            raise ValueError(
                "No features were selected for analysis"
            )

        # ---------------------------------------------------------
        # 2. Update status
        # ---------------------------------------------------------

        update_analysis_progress(
            self.analysis_request_id,
            10,
            "Loading product reviews",
        )

        # ---------------------------------------------------------
        # 3. Load reviews from Supabase
        # ---------------------------------------------------------

        reviews = get_product_reviews(product_id)

        if not reviews:
            raise ValueError(
                "No reviews are available for this product"
            )

        print(
            f"Loaded {len(reviews)} reviews for analysis"
        )

        # ---------------------------------------------------------
        # 4. Convert Supabase reviews to analysis input
        # ---------------------------------------------------------
        #
        # We deliberately DO NOT use Pandas or the old CSV-style
        # fields such as reviews.text / reviews.rating.
        #
        # Supabase already gives us normalized fields:
        #   rating
        #   title
        #   content
        #   verified
        #   helpful_votes
        # ---------------------------------------------------------

        review_dicts = []

        for review in reviews:
            review_dicts.append({
                "rating": review.get("rating"),
                "title": review.get("title") or "",
                "text": review.get("content") or "",
                "verified": review.get("verified", False),
                "helpful_votes": review.get("helpful_votes") or 0,
            })

        review_dicts = [
            review
            for review in review_dicts
            if review["text"].strip()
        ]

        if not review_dicts:
            raise ValueError(
                "No usable review content is available"
            )

        # ---------------------------------------------------------
        # 5. Initialize CrewAI team
        # ---------------------------------------------------------

        team = TeamRevify()

        # ---------------------------------------------------------
        # 6. Summarize reviews in chunks
        # ---------------------------------------------------------

        update_analysis_progress(
            self.analysis_request_id,
            30,
            f"Summarizing {len(review_dicts)} reviews",
        )

        chunk_summaries = summarize_reviews_chunked(
            review_dicts,
            team,
            chunk_size=50,
        )

        if not chunk_summaries:
            raise ValueError(
                "Review summarization produced no output"
            )

        reviews_input = "\n\n".join(chunk_summaries)

        # ---------------------------------------------------------
        # 7. Final feature-based analysis
        # ---------------------------------------------------------

        update_analysis_progress(
            self.analysis_request_id,
            60,
            "Running feature-based review analysis",
        )

        review_agent = team.review_analysis_agent()

        analysis_task = (
            team.comprehensive_review_analysis_task()
        )

        analysis_crew = Crew(
            agents=[review_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False,
        )

        result = analysis_crew.kickoff(
            inputs={
                "features": ", ".join(selected_features),
                "reviews": reviews_input,
            }
        )

        # ---------------------------------------------------------
        # 8. Read structured output
        # ---------------------------------------------------------

        if result.pydantic:
            analysis_result = result.pydantic.model_dump()

        elif result.json_dict:
            analysis_result = result.json_dict

        else:
            raise ValueError(
                "Final analysis did not return structured output"
            )

        # ---------------------------------------------------------
        # 9. Persist final result
        # ---------------------------------------------------------

        update_analysis_progress(
            self.analysis_request_id,
            90,
            "Saving analysis results",
        )

        final_result = {
            "features": selected_features,
            "analysis": analysis_result,
            "total_reviews": len(review_dicts),
        }

        update_analysis_request(
            self.analysis_request_id,
            status="completed",
            result=final_result,
        )

        update_analysis_progress(
            self.analysis_request_id,
            100,
            "Analysis completed successfully",
        )

        print(
            f"✅ Analysis completed for "
            f"{self.analysis_request_id}"
        )

        return final_result