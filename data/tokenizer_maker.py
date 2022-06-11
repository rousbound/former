# from transformers import AutoModel, AutoTokenizer

# BERT Base
# tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

# print(tokenizer("Olá pessoal"))
# print(tokenizer.decode(1651))

from tokenizers import ByteLevelBPETokenizer
import os
# initialize
tokenizer = ByteLevelBPETokenizer()
# and train

tokenizer.train(files="portadosfundos_filtered.txt", vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


# try:
    # os.rmdir('portificador')
# except:
    # pass
# os.mkdir('portificador')


tokenizer.save_model('portificador')

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('portificador')

print(tokenizer("Olá pessoal, meu nome é Jorge")['input_ids'])
print(tokenizer("Olá pessoal, meu nome é Jorge")['input_ids'])
print(tokenizer("Olá")['input_ids'])
for tok in tokenizer("Olá pessoal, meu nome é Jorge")['input_ids']:
    print(tok, tokenizer.decode(tok))

# with open("portadosfundos_filtered.txt") as file:
    # from transformers import RobertaTokenizerFast
    # nbatches = 100000
    # n_train = int(nbatches*0.9) 
    # n_valid = int(nbatches*0.05)
    # n_test = int(nbatches*0.05)
    # tokenizer = RobertaTokenizerFast.from_pretrained('portificador')
    # # X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
    # X = tokenizer(file.read(n_train + n_valid + n_test))['input_ids']
    # print(X)
    # trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    # print("trX:",trX)
    # print("vaX:",vaX)
    # print("teX:",teX)

# with open("portadosfundos_filtered.txt", "r") as arq:
# for line in arq.readlines():
    # if line.strip() != "":
        # print(line.strip())
        # for word

