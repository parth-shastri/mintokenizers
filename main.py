from src.tokenizers.gpt4o_tokenizer import GPT4OTokenizer
import os


if __name__ == '__main__':
    tokenizer = GPT4OTokenizer()
    save_dir = "./models/gpt4o_tokenizer/"

    os.makedirs(save_dir, exist_ok=True)

    tokenizer.save(file_prefix=os.path.join(save_dir, "gpt4o_tokenizer"))
