from crewai import Crew, Process
from crewai.flow.flow import Flow, and_, listen, start

from ..crews.team_revify.team_revify import TeamRevify
from ..services.analysis_service import (
    set_extracted_features,
    update_analysis_progress,
    update_analysis_request,
)
from ..services.zebu_review_service import (
    fetch_and_persist_reviews,
)
from ..utils.amazon import extract_asin


class RevifyPreparationFlow(Flow):

    def __init__(
        self,
        analysis_request_id: str,
        product_url: str,
        product_name: str,
    ):
        super().__init__()

        self.analysis_request_id = analysis_request_id
        self.product_url = product_url
        self.product_name = product_name

    @start()
    def extract_features(self):
        update_analysis_progress(
            self.analysis_request_id,
            20,
            "Extracting product features",
        )

        team = TeamRevify()

        crew = Crew(
            agents=[
                team.feature_extractor()
            ],
            tasks=[
                team.extract_features_task()
            ],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff(
            inputs={
                "product_input": self.product_url
            }
        )

        if not result.pydantic:
            raise RuntimeError(
                "Feature extraction did not return structured output"
            )

        features = result.pydantic.features

        set_extracted_features(
            self.analysis_request_id,
            features,
        )

        update_analysis_progress(
            self.analysis_request_id,
            50,
            "Product features extracted",
        )

        return features

    @start()
    def acquire_reviews(self):
        update_analysis_progress(
            self.analysis_request_id,
            20,
            "Acquiring Amazon reviews",
        )

        asin = extract_asin(self.product_url)

        if not asin:
            raise ValueError(
                "Could not extract ASIN from Amazon URL"
            )

        reviews = fetch_and_persist_reviews(
            asin=asin,
            product_url=self.product_url,
            product_name=self.product_name,
        )

        update_analysis_progress(
            self.analysis_request_id,
            70,
            f"{len(reviews)} reviews acquired",
        )

        return reviews

    @listen(and_(extract_features, acquire_reviews))
    def preparation_complete(self):
        update_analysis_request(
            self.analysis_request_id,
            status="awaiting_selection",
        )

        update_analysis_progress(
            self.analysis_request_id,
            100,
            "Ready for feature selection",
        )

        return {
            "status": "awaiting_selection"
        }