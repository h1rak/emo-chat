from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class Strategy:
    def __init__(self):
        self.strategy_history = ""
        self.history = []
        self.model_name = "/workspace/santa/ESC-chat/checkpoint-14523"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

        # 念のため PAD/EOS を明示
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # 履歴の区切り
        self.sep = self.tokenizer.eos_token or self.tokenizer.sep_token or "</s>"

    def pred_strategy(self, prompt: str) -> str:
        self.history.append(prompt)
        # 履歴に追記（区切りを入れて貯める）
        self.strategy_history = (
            f"{self.strategy_history}{self.sep}{prompt}" if self.strategy_history else prompt
        )

        self.model.eval()
        inputs = self.tokenizer(self.strategy_history, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 次のプロンプトに活かせるよう履歴へも追記
        self.strategy_history += f"{self.sep}{decoded_output}"
        print(self.strategy_history)

        return decoded_output

