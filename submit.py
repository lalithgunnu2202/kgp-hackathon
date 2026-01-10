import pandas as pd

from src.load_data import load_train_test, load_novel
from src.chunker import chunk_text
from src.claims import extract_claims
from src.retriever import SemanticRetriever
from src.decision import decide_claim, final_decision


def predict_sample(sample, retriever, chunks):
    backstory = sample["content"]
    claims = extract_claims(backstory)

    claim_results = []

    for claim in claims:
        results = retriever.retrieve(claim, top_k=3)
        best_score = max(score for _, score in results)

        is_supported = decide_claim(
            best_score,
            claim=claim,
            evidence_chunks=[text for text, _ in results]
        )

        claim_results.append(is_supported)

    return final_decision(claim_results)



def main():
    _, test_df = load_train_test()

    predictions = []

    # Process test set row by row
    for idx, row in test_df.iterrows():
        novel_name = row["book_name"]

        novel_text = load_novel(novel_name)
        chunks = chunk_text(novel_text, chunk_size=1000, overlap=200)

        retriever = SemanticRetriever()
        retriever.index_chunks(chunks)

        pred = predict_sample(row, retriever, chunks)
        predictions.append(pred)

        print(f"Processed sample {idx} → {pred}")

    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": predictions
    })

    submission.to_csv("submission.csv", index=False)
    print("\nsubmission.csv generated successfully!")


if __name__ == "__main__":
    main()
