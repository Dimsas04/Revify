import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_output/ablation_scores.csv")
df = df.sort_values("chunk_size")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Ablation Study: Chunk Size vs Metrics", fontsize=14, fontweight="bold")

# ── NLP Scores ──────────────────────────────────────────────
ax = axes[0]
ax.plot(df["chunk_size"], df["rouge1_f1"], marker="o", label="ROUGE-1")
ax.plot(df["chunk_size"], df["rouge2_f1"], marker="o", label="ROUGE-2")
ax.plot(df["chunk_size"], df["rougeL_f1"], marker="o", label="ROUGE-L")
ax.plot(df["chunk_size"], df["bert_f1"],   marker="o", label="BERT F1")
ax.set_title("NLP Scores")
ax.set_xlabel("Chunk Size")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True)

# ── Total Time ───────────────────────────────────────────────
ax = axes[1]
ax.bar(df["chunk_size"], df["summarization_time"], label="Summarization", color="steelblue")
ax.bar(df["chunk_size"], df["analysis_time"], bottom=df["summarization_time"], label="Analysis", color="coral")
ax.set_title("Time Breakdown (s)")
ax.set_xlabel("Chunk Size")
ax.set_ylabel("Seconds")
ax.legend()
ax.grid(True, axis="y")

# ── Tokens Estimate ──────────────────────────────────────────
ax = axes[2]
ax.plot(df["chunk_size"], df["tokens_estimate"], marker="s", color="green")
ax.set_title("Token Estimate (final prompt)")
ax.set_xlabel("Chunk Size")
ax.set_ylabel("Tokens")
ax.grid(True)

plt.tight_layout()
plt.savefig("benchmark_output/ablation_plot.png", dpi=150)
plt.show()
print("Saved: benchmark_output/ablation_plot.png")
