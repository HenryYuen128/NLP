import torch
from transformers import AutoTokenizer, AutoConfig
from models.TextClassification import BertFamily

device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained('/home/henry/code/NLP/hotel_sentiment/hfl-chinese-roberta-wwm-ext_bert_BCE_mean_NO_2023-09-08_15:17:18/checkpoint-245')
train_encodings = tokenizer('娃哈哈', max_length=128, truncation=True, padding='max_length', return_token_type_ids=True, return_tensors='pt')
# train_encodings.to(device)
# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
# config = BertConfig(
#     vocab_size_or_config_json_file=32000,
#     hidden_size=768,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     intermediate_size=3072,
#     torchscript=True,
# )

# Instantiating the model
model = BertFamily.BertVanillaTextClsForTransformers.from_pretrained('/home/henry/code/NLP/hotel_sentiment/hfl-chinese-roberta-wwm-ext_bert_BCE_mean_NO_2023-09-08_15:17:18/checkpoint-245')
model.to(device)
# The model needs to be in evaluation mode
model.eval()


# Creating the trace
traced_model = torch.jit.trace(model, [train_encodings['input_ids'].to(device), train_encodings['token_type_ids'].to(device), train_encodings['attention_mask'].to(device)])
torch.jit.save(traced_model, "traced_bert.pt")