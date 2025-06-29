from flask import Flask, render_template, request
import pandas as pd
from bm25_approach import find_matching_articles

app = Flask(__name__)

# === Load dataset dan kategori ===
data = pd.read_csv("klikdokter_articles_multi_category.csv")
categories = sorted(data['category'].dropna().unique().tolist())

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    selected_category = request.args.get("category", "")

    # Kalau POST (ada query dari form pencarian)
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            all_results = find_matching_articles(query, top_k=20)

            # Filter by kategori jika dipilih
            if selected_category:
                results = [r for r in all_results if r["category"] == selected_category]
            else:
                results = all_results

    # Kalau GET dan hanya klik kategori
    elif selected_category:
        filtered = data[data["category"] == selected_category][['title', 'content', 'link', 'category']]
        filtered = filtered.fillna("")
        results = []
        for _, row in filtered.iterrows():
            results.append({
                "title": row["title"],
                "content": row["content"],
                "link": row["link"],
                "category": row["category"],
                "score": 0.0,
                "confidence": 0.0
            })

    return render_template(
        "index.html",
        query=query,
        results=results,
        categories=categories,
        selected_category=selected_category
    )

if __name__ == "__main__":
    app.run(debug=True)
