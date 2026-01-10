from src.load_data import load_train_test, load_novel
from src.chunker import chunk_text
from src.claims import extract_claims
from src.retriever import SemanticRetriever
from src.decision import decide_claim, final_decision
import pandas as pd
from logger import logger
def main():
    # train_df, _ = load_train_test()
    for high,low in [(0.42,0.25),(0.41,0.25),(0.43,0.25),(0.44,0.25),(0.46,0.25)]:
        accurate = 0
        false=0
        train_df = pd.read_csv("data/train.csv")
        for num in [15,17,18,21,22,26,30,12,10,8,56,40,80,73,74,39,40,41]:
            sample = train_df[num]

            novel_name = sample["book_name"]
            character = sample["char"]
            backstory = sample["content"]
            true_label = sample["label"]

            print("Novel:", novel_name)
            print("Character:", character)
            print("True label:", true_label)

            # load and chunk novel
            novel_text = load_novel(novel_name)
            chunks = chunk_text(novel_text, chunk_size=1000, overlap=200)

            # index chunks
            retriever = SemanticRetriever()
            retriever.index_chunks(chunks)

            # extract claims
            claims = extract_claims(backstory)

            print("\n--- CLAIM ANALYSIS ---\n")

            claim_results = []

            for claim in claims:
                results = retriever.retrieve(claim, top_k=3)
                best_score = max(score for _, score in results)

                is_supported = decide_claim(
                    best_score,
                    claim=claim,
                    evidence_chunks=[text for text, _ in results],
                    high_thres=high,
                    low_thre=low
                )

                claim_results.append(is_supported)

                print("CLAIM:", claim)
                print("Best similarity score:", round(best_score, 4))
                print("Supported:", is_supported)
                print("-" * 50)

            # final decision
            predicted_label = final_decision(claim_results)
            
            print("\n=== FINAL DECISION ===")
            print("Predicted label:", predicted_label)
            print("True label     :", true_label)
            if predicted_label==true_label:
                accurate+=1
            elif predicted_label!=true_label:
                false+=1
        accuracy = accurate/(accurate+false)
        logger.info(f"high={high}  low={low}  accurracy = {accuracy}")

if __name__ == "__main__":
    main()

