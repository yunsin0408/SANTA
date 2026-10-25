from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, Seq2SeqTrainingArguments

@dataclass
class SantaArguments:
    l_max_len: int = field(
        default=35,
        metadata={
            "help": "The max length of label"}
    )
    use_generate: bool = field(
        default=True,
        metadata={"help": "Whether to get generate loss"}
    )
    use_contrastive: bool = field(
        default=True,
        metadata={"help": "Whether to include the SDA in-batch contrastive loss. False = MEP-only pretraining "
                          "(needs --use_generate True); keep True for finetuning and every other run."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Divides the contrastive scores; use with --normalize True (cosine), e.g. 0.05"}
    )
    similarity: str = field(
        default="dot",
        metadata={"help": "'dot' (inner product; cosine with --normalize True) or 'poincare' (negative Poincare-ball "
                          "distance, after mapping reps into the ball; needs --normalize False). Saved in the checkpoint."}
    )
    poincare_scale: float = field(
        default=0.25,
        metadata={"help": "Reps are multiplied by this before the tanh map into the Poincare ball. Mean-pooled CodeT5 "
                          "reps have norm ~3-4, where tanh saturates (every point on the boundary); 0.25 maps them to "
                          "radius ~0.66-0.76. Saved in the checkpoint."}
    )
