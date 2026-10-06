# Project structure

Revify is a full-stack project: a Flask API and CrewAI/Python workflow under `revify_flow`, plus a React/Vite frontend under `revify-frontend`.

## Scraping path

- `revify_flow/src/revify_flow/api.py` exposes asynchronous-looking HTTP endpoints and owns process-global job status.
- `revify_flow/src/revify_flow/tools/amazon_scraper_tool.py` is the active scraper implementation.
- `revify_flow/src/revify_flow/crews/team_revify/team_revify.py` injects the scraper into the CrewAI review-scraper agent.
- The frontend polls `/api/feature-status` and `/api/status` every two seconds.
- The scraper and API communicate through `scraped_reviews.csv`, not a job-scoped repository.

## Functional areas

1. Feature extraction
2. Amazon review scraping
3. Chunked LLM summarization and feature analysis
4. Flask status/result/download API
5. React progress and result presentation

The repository also contains several experimental Selenium scrapers; only the `AmazonScraperTool` path is wired into the CrewAI/API flow.
