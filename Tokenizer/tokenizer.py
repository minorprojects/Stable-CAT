#please uncomment to try out code. The code is for training yout own tokenizer using wikitext_103

# !wget http://www.gutenberg.org/cache/epub/16457/pg16457.txt
# !wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
# !unzip wikitext-103-raw-v1.zip

# import tokenizer
# !pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
unk_token = "<UNK>"
spl_tokens = ["<UNK>", "<SEP>","<MASK>","<CLS>"]

# def token_trainer(alg):
#     tokenizer = Tokenizer(BPE(unk_token=unk_token))
#     trainer= BpeTrainer(special_tokens = spl_tokens)
#     tokenizer.pre_tokenizer = Whitespace()
#     return tokenizer,trainer

# def train_tokenizer(files,alg='BPE'):
#     tokenizer,trainer = token_trainer(alg)
#     tokenizer.train(files,trainer)
#     tokenizer.save("./peregrine_tokenizer.json")
#     tokenizer = Tokenizer.from_file("./peregrine_tokenizer.json")
#     return tokenizer

def tokenize(input_string,tokenizer):
    output = tokenizer.encode(input_string)
    return output

# small_file = ['/kaggle/working/pg16457.txt']
# large_files = [f"/kaggle/working/wikitext-103-raw/wiki.{split}.raw" for split in ['test','train','valid']]
# for files in [small_file,large_files]:
#     trained_tokenizer = train_tokenizer(files,"BPE")
#     input_string = "This is something cool and i love it so much tokenization"
#     output =  tokenize(input_string,trained_tokenizer)
# #     tokens_dict[alg] = output.tokens
#     print(output.tokens,'->',len(output.tokens))
input_string = "This is something cool and i love it so much tokenization"
tokenizer = Tokenizer.from_file("/kaggle/working/peregrine_tokenizer.json")
output = tokenize(input_string,tokenizer)
print(output.ids)
tokenizer.decode(output.ids)
