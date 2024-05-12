from transformers import AutoModel


def is_channel_id_provided(channel_ids: list[str]) -> bool:
    for id in channel_ids:
        if id:
            return True
    
    return False

def get_model(token: str = "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"):
    return AutoModel.from_pretrained("DeepPavlov/rubert-base-cased", token=token)