# Tech stack

- Python `>=3.10, <=3.13` for the `revify_flow` package.
- Flask with Flask-CORS for the HTTP API.
- CrewAI with Gemini-backed agents for feature extraction, scraping orchestration, and analysis.
- Selenium `4.29.0`, `undetected_chromedriver` in the active scraper, and BeautifulSoup `4.13.3`.
- pandas for CSV persistence and review transformation.
- React 18/Vite frontend with Axios polling the Flask API.
- Environment variables `NUMBER` and `PASSWORD` are read for Amazon login.

The active scraper is synchronous internally and uses fixed/random sleeps, browser page-source snapshots, XPath/CSS selectors, and a single fixed output filename. The package manifest shown in `revify_flow/pyproject.toml` declares only `crewai[tools]`; the root requirements file contains the broader runtime dependencies.
