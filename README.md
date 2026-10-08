# Revify: Agentic Customer Review Analysis

Revify is an advanced, agentic-architecture-powered project that automates the analysis of customer reviews using a multi-agent system built atop [crewAI](https://crewai.com). Designed for flexibility and extensibility, Revify enables deep, feature-based sentiment analysis, efficient review scraping, and collaborative research generation, all orchestrated through autonomous, specialized AI agents.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Agentic Architecture](#agentic-architecture)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)
- [Support](#support)

---

## Features

- **Multi-Agent AI Workflow**: Employs autonomous agents for orchestrated review scraping, feature extraction, and analysis.
- **Amazon Review Scraping**: Automated extraction of customer reviews from Amazon product pages, with robust CSV export.
- **Feature Extraction**: Identifies and extracts key product features from reviews (e.g., Comfort, Durability, Battery Life, etc.).
- **Sentiment Analysis & Reporting**: Structured output of feature-based sentiment and insights, ready for research or product evaluation.
- **Extensible CrewAI Framework**: Easily add, configure, and coordinate agents for new domains or research tasks.
- **Debug & Logging Utilities**: Built-in debug functions and extensive logging for workflow transparency and troubleshooting.

---

## Agentic Architecture

Revify utilizes an agentic paradigm via CrewAI, where each agent fulfills a specialized role:

- **Review Scraper Agent**
  - Scrapes reviews from Amazon using Selenium and BeautifulSoup.
  - Saves structured data (title, text, rating, etc.) to CSV.
- **Feature Extraction Agent**
  - Analyzes review texts to extract salient product features.
  - Supports JSON-based output for downstream analysis.
- **Sentiment Analysis Agent**
  - Assigns sentiment scores to extracted features.
  - Aggregates insights for report generation.
- **Task Orchestration**
  - Defined in `config/tasks.yaml` and `config/agents.yaml` for each flow.
  - Agents collaborate sequentially or in parallel as configured.

**Sample Extracted Features:**
- Comfort, Durability, Fit, Support, Material Quality, Breathability, Traction, Weight, Style, Value for Money
- Display Quality, Battery Life, Storage Capacity, Portability, User Interface, Connectivity, Price

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dimsas04/MajorProj.git
   cd MajorProj
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Python 3.9+ required. See `requirements.txt` for full dependency list (CrewAI, LangChain, Selenium, Pydantic, etc.)*

---

## Running the Project

To launch an agent flow (e.g., `revify_flow`):

```bash
crewai run
```

This command initializes the agent crew, triggering the full pipeline from review scraping to feature-based report generation.

- The system will create a `report.md` summarizing findings in the root directory.
- For custom tasks or agents, modify `config/tasks.yaml` and `config/agents.yaml`.

---

## Configuration

- **Agents & Tasks:**  
  Configure agent roles, skills, and task assignments in:
  - `revify_flow/config/agents.yaml`
  - `revify_flow/config/tasks.yaml`

- **Debugging:**  
  Dedicated debug functions (`debug_scraper_tool`, `debug_run_workflow`) in `revify_flow/src/revify_flow/main.py` allow step-by-step validation and logging.

---

## Technologies Used

- **[CrewAI](https://crewai.com):** Multi-agent orchestration framework
- **Python Ecosystem:**  
  - Data: pandas, numpy
  - AI: langchain, pydantic, langgraph, openai
  - Web Scraping: selenium, beautifulsoup4
- **Logging & Debugging:** Python logging, tracebacks
- **Configuration:** YAML, dotenv
- **Other:** Google Cloud, Matplotlib, OpenCV (for future extensibility)

---

## Support

## Supabase setup

Revify uses Supabase for authentication and PostgreSQL persistence while Flask
continues to own the CrewAI/Gemini workflow.

1. Create a Supabase project and copy its project URL.
2. Copy the **publishable** API key to `revify-frontend/.env` as
   `VITE_SUPABASE_PUBLISHABLE_KEY`, and the URL as `VITE_SUPABASE_URL`.
3. Copy the **secret** API key only to `revify_flow/.env` as
   `SUPABASE_SECRET_KEY`, with the same URL as `SUPABASE_URL`. Never expose or
   commit this key.
4. Apply `supabase/migrations/20261006000000_revify_schema.sql` with the
   Supabase CLI (`supabase db push`) or the SQL editor.
5. In Supabase Authentication > URL Configuration, set the local Site URL to
   `http://localhost:5173` and add it to Redirect URLs.
6. In Authentication > Providers > Google, configure a Google OAuth client.
   The Google callback URI is the Supabase dashboard's
   `https://<project-ref>.supabase.co/auth/v1/callback` URL; copy that exact
   value into Google Cloud OAuth credentials. Do not put the Google client
   secret in this repository.
7. Install dependencies and start the applications:

   ```bash
   cd revify-frontend
   npm install
   npm run dev

   cd ../revify_flow
   pip install -e .
   python -m src.revify_flow.api
   ```

The frontend publishable key is safe for browser use because database access is
controlled by RLS. The backend secret key is server-only and bypasses RLS for
trusted persistence operations. Email/password signup, email/password login,
Google login, session restoration, logout, and protected `/analysis` and
`/results` routes are implemented in the frontend. The profile trigger creates
one application profile for every Supabase Auth user.

The backend validates the `Authorization: Bearer <access-token>` header with
Supabase before user-specific operations and derives the user ID from the
validated token rather than trusting a request-body user ID. Analysis history
is available at `GET /api/history`.

### Persisting Apify reviews

`apify/apifyScrapper.py` now persists the Zebu actor dataset after collection.
It upserts one `products` row using the ASIN and `last_scraped_at`, with
`category` and `brand` set to `Unknown` until feature extraction supplies those
values. It then upserts normalized review rows using the schema's
`(product_id, source, external_review_id)` uniqueness constraint. Missing review
IDs receive a deterministic hash-based ID, and records without review content
are skipped because `reviews.content` is required.

The Flask analytics pipeline uses the reusable
`revify_flow/src/revify_flow/tools/apifyScrapper_tool.py` integration. It looks
up the product by ASIN and loads its reviews from Supabase first. Apify is
called only when no reviews exist for that product; newly acquired reviews are
persisted before feature summarization and analysis. The final feature-based
JSON is saved in the matching `analysis_requests.result` column, with the
request status updated to `completed` or `failed`.

Run it from the repository root after setting `SUPABASE_URL`,
`SUPABASE_SECRET_KEY`, and `APIFY_API_TOKEN` in the local environment:

```bash
python apify/apifyScrapper.py
```

For questions, feedback, or to contribute:

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
- [Join CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with CrewAI Docs](https://chatg.pt/DWjSBZn)

---

Let's create wonders together with the power and simplicity of agentic AI!
