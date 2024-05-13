from torch.utils.data import Dataset

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = BertModel.from_pretrained(
    "DeepPavlov/rubert-base-cased", output_hidden_states=True
)


class CustomDataset(Dataset):
    def __init__(self, X):
        self.text = X

    def tokenize(self, text):
        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=150,
        )

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        output = self.text[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}
