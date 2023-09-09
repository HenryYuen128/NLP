import torch
from transformers import AutoTokenizer, AutoConfig
from models.TextClassification import BertFamily
import torch_tensorrt

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
# model = BertFamily.BertVanillaTextClsForTransformers.from_pretrained('/home/henry/code/NLP/hotel_sentiment/hfl-chinese-roberta-wwm-ext_bert_BCE_mean_NO_2023-09-08_15:17:18/checkpoint-245')
model = torch.jit.load('/home/henry/code/NLP/traced_bert.pt')
model.to(device)
# The model needs to be in evaluation mode
model.eval()


# # Creating the trace
# traced_model = torch.jit.trace(model, [train_encodings['input_ids'].to(device), train_encodings['token_type_ids'].to(device), train_encodings['attention_mask'].to(device)])
# torch.jit.save(traced_model, "traced_bert.pt")


# input_signature expects a tuple of individual input arguments to the module
# The module below, for example, would have a docstring of the form:
# def forward(self, input0: List[torch.Tensor], input1: Tuple[torch.Tensor, torch.Tensor])
batch_size=128

# dynamic shape
inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 128],
        opt_shape=[batch_size, 128],
        max_shape=[batch_size, 128],
        dtype=torch.int64,
    ),
        torch_tensorrt.Input(
        min_shape=[1, 128],
        opt_shape=[batch_size, 128],
        max_shape=[batch_size, 128],
        dtype=torch.int64,
    ),
        torch_tensorrt.Input(
        min_shape=[1, 128],
        opt_shape=[batch_size, 128],
        max_shape=[batch_size, 128],
        dtype=torch.int64,
    )
]
# enabled_precisions = {torch.float, torch.half}  # Run with fp16
enabled_precisions= {torch.float32} # run with 32-bit precision

trt_model_fp16 = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions,
    truncate_long_and_double=True
)


# trt_model_fp16 = torch_tensorrt.compile(model, 
#     inputs= [torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int64),  # input_ids
#              torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int64),  # token_type_ids
#              torch_tensorrt.Input(shape=[batch_size, 128], dtype=torch.int64)], # attention_mask
#     enabled_precisions= {torch.half}, # Run with 16-bit precision
#     truncate_long_and_double=True
# )


input_data = train_encodings.to("cuda")
with torch.no_grad():
    result = trt_model_fp16(input_data['input_ids'], input_data['token_type_ids'], input_data['attention_mask'])
    torch.jit.save(trt_model_fp16, "trt_ts_module_dynamic_shape_32bit.ts")