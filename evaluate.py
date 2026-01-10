from src.load_data import load_train_test, load_novel
from src.chunker import chunk_text
from src.claims import extract_claims
from src.retriever import SemanticRetriever
from src.decision import decide_claim, final_decision


def predict(sample):
    novel_text = load_novel(sample["book_name"])
    chunks = chunk_text(novel_text, chunk_size=1000, overlap=200)

    retriever = SemanticRetriever()
    retriever.index_chunks(chunks)

    claims = extract_claims(sample["content"])
    claim_results = []

    for claim in claims:
        results = retriever.retrieve(claim, top_k=3)
        best_score = max(score for _, score in results)

        supported = decide_claim(
            best_score,
            claim=claim,
            evidence_chunks=[text for text, _ in results]
        )
        claim_results.append(supported)

    return final_decision(claim_results)


def main():
    train_df, _ = load_train_test()

    correct = 0
    total = len(train_df)

    for idx, row in train_df.iterrows():
        pred_bool = predict(row)
        true_bool = (row["label"] == "consistent")

        if pred_bool == true_bool:
            correct += 1

        print(
            f"Sample {idx}: "
            f"Predicted={'consistent' if pred_bool else 'inconsistent'}, "
            f"True={row['label']}"
        )

    accuracy = correct / total
    print("\n========================")
    print(f"Accuracy on TRAIN set: {correct}/{total} = {accuracy:.3f}")
    print("========================")


if __name__ == "__main__":
    main()
