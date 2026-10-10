from ..crews.team_revify.team_revify import TeamRevify
from crewai import Crew, Process, Task

def summarize_reviews_chunked(review_data, team, chunk_size):
    print(f"\n🔧 Chunking and summarizing {len(review_data)} reviews...")
    
    review_chunks = [
        review_data[i:i + chunk_size]
        for i in range(0, len(review_data), chunk_size)
    ]
    summaries = []
    team = TeamRevify()
    summarize_agent = team.chunk_summary_agent()  # you’ll define this in YAML
    print(f"🧠 Loaded summary agent to handle {len(review_chunks)} chunks")

    for i, chunk in enumerate(review_chunks):
        print(f"\n📝 Summarizing chunk {i+1}/{len(review_chunks)}...")
        task = Task(
            description=(
                "Summarize the following list of product reviews. Focus on overall tone, frequently mentioned features, "
                "and any strong sentiments. This is just one chunk of many.\n\n"
                f"Reviews:\n{chunk}"
            ),
            expected_output="A concise paragraph summarizing this chunk of reviews.",
            agent=summarize_agent
        )

        crew = Crew(
            agents=[summarize_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        summaries.append(result.raw)

    print(f"\n✅ Done summarizing all chunks. Total summaries: {len(summaries)}")
    return summaries