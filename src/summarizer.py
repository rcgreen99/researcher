import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Summarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_ckpt = "google/pegasus-cnn_dailymail"
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(self.device)

    def summarize(self, text):
        tokenized_input = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        summary = self.model.generate(
            input_ids=tokenized_input["input_ids"].to(self.device),
            attention_mask=tokenized_input["attention_mask"].to(self.device),
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
        )

        decoded_summary = self.tokenizer.decode(
            summary[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        decoded_summary = decoded_summary.replace("<n>", " ")

        return decoded_summary


if __name__ == "__main__":
    test_article = """
    Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom')[1][2] is the systematized study of general and fundamental questions, such as those about existence, reason, knowledge, values, mind, and language.[3][4][5][6][7] Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE),[8][9] although this theory is disputed by some.[10][11][12] Philosophical methods include questioning, critical discussion, rational argument, and systematic presentation.[13][14][i]

    Historically, philosophy encompassed all bodies of knowledge and a practitioner was known as a philosopher.[15] "Natural philosophy", which began as a discipline in ancient India and Ancient Greece, encompasses astronomy, medicine, and physics.[16][17] For example, Isaac Newton's 1687 Mathematical Principles of Natural Philosophy later became classified as a book of physics. In the 19th century, the growth of modern research universities led academic philosophy and other disciplines to professionalize and specialize.[18][19] Since then, various areas of investigation that were traditionally part of philosophy have become separate academic disciplines, and namely the social sciences such as psychology, sociology, linguistics, and economics.

    Today, major subfields of academic philosophy include metaphysics, which is concerned with the fundamental nature of existence and reality; epistemology, which studies the nature of knowledge and belief; ethics, which is concerned with moral value; and logic, which studies the rules of inference that allow one to derive conclusions from true premises.[20][21] Other notable subfields include philosophy of religion, philosophy of science, political philosophy, aesthetics, philosophy of language, and philosophy of mind.  
    """

    summarizer = Summarizer()

    summary = summarizer.summarize(test_article)

    print(summary)
