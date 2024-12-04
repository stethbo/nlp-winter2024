from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import torch
from datasets import load_dataset
import pandas as pd


class Retriever:
    def __init__(self, model_ckpt):
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = None

    def add_dataset_and_index(self, dataset: Dataset, document_column: str):
        embeddings_dataset = dataset.map(
            lambda x: {
                "embeddings": self._encode(x[document_column]).detach().cpu().numpy()[0]
            }
        )
        embeddings_dataset.add_faiss_index(column="embeddings")
        self.dataset = embeddings_dataset

    def find_similar(self, query: str, top_k: int = 5):
        self._check_dataset_exists()

        query_embedding = self._encode(query).detach().cpu().numpy()
        results = self.dataset.get_nearest_examples(
            "embeddings", query_embedding, k=top_k
        )
        return results

    def _encode(self, text_list: list) -> torch.Tensor:
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self._cls_pooling(model_output)

    # We get the [CLS] token hidden state from the transformer output and treat it as the text embedding
    def _cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def _check_dataset_exists(self):
        if self.dataset is None:
            raise ValueError("Please add a dataset and create the index first.")


if __name__ == "__main__":
    # create retriever object
    retriever = Retriever("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # The following code block is for creating dataset and is copied from the notebook
    issues_dataset = load_dataset("lewtun/github-issues", split="train")
    issues_dataset = issues_dataset.filter(
        lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
    )
    columns = issues_dataset.column_names
    columns_to_keep = ["title", "body", "html_url", "comments"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    issues_dataset = issues_dataset.remove_columns(columns_to_remove)
    issues_dataset.set_format("pandas")
    df = issues_dataset[:]
    comments_df = df.explode("comments", ignore_index=True)
    comments_dataset = Dataset.from_pandas(comments_df)
    comments_dataset = comments_dataset.map(
        lambda x: {"comment_length": len(x["comments"].split())}
    )
    comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

    def concatenate_text(examples):
        return {
            "text": examples["title"]
            + " \n "
            + examples["body"]
            + " \n "
            + examples["comments"]
        }

    comments_dataset = comments_dataset.map(concatenate_text)

    # plug in dataset
    retriever.add_dataset_and_index(comments_dataset, "text")

    # Retrieval using the plugged dataset
    scores, samples = retriever.find_similar(
        "How can I load a dataset offline?", top_k=5
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()
