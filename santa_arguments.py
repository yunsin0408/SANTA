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
        metadata={"help": "'dot' (inner product; cosine with --normalize True), 'poincare' (negative Poincare-ball "
                          "distance, after mapping reps into the ball) or 'lorentz' (negative hyperboloid distance, after "
                          "mapping reps onto the hyperboloid); the last two need --normalize False. Saved in the checkpoint."}
    )
    poincare_scale: float = field(
        default=0.25,
        metadata={"help": "Reps are multiplied by this before the tanh map into the Poincare ball. Mean-pooled CodeT5 "
                          "reps have norm ~3-4, where tanh saturates (every point on the boundary); 0.25 maps them to "
                          "radius ~0.66-0.76. Saved in the checkpoint."}
    )
    lorentz_scale: float = field(
        default=0.5,
        metadata={"help": "Reps are multiplied by this to give their hyperbolic radius on the hyperboloid. 0.5 gives the "
                          "same distances as --similarity poincare with --poincare_scale 0.25, whose tanh map doubles "
                          "the radius. Saved in the checkpoint."}
    )
