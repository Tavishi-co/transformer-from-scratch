import torch
from src.transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 256
SEQ_LEN = 6

BOS = 1

model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=SEQ_LEN + 5
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "transformer_toy.pt",
        map_location=DEVICE
    )
)

src = torch.tensor(
    [[3, 7, 12, 5, 18, 9]],
    dtype=torch.long
)

prediction = model.generate(
    src,
    max_len=SEQ_LEN,
    bos_token_id=BOS
)

expected = torch.flip(src, dims=[1])

print("Input:     ", src.tolist())
print("Expected:  ", expected.tolist())
print("Predicted: ", prediction.tolist())