from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


def get_vocab_sim(first_vocab_keys,second_vocab_keys):
    f_set=set(first_vocab_keys)
    s_set=set(second_vocab_keys)
    intersection = f_set.intersection(s_set)
    union = f_set.union(s_set)

    similarity_jaccard= len(intersection)/len(union)
    vocab_f = f_set-s_set
    return intersection, len(intersection)/len(f_set), len(intersection)/len(s_set),similarity_jaccard,vocab_f




def get_training_corpus(dataset):
    for i in range(0, len(dataset["train"]),1000):
        yield dataset["train"][i:i+1000]["texte"]
def main(args):
    data_files={"train":args.train_dataset,"test":args.test_dataset,"valid":args.valid_dataset}
    swerick_dataset = load_dataset("pandas",data_files=data_files)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False,strip_accents=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=50325, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(swerick_dataset), trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.save(args.tokenizer_file)








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dataset", type=str, default="swerick_data_random_train.pkl", help=" pkl file of the train dataset")
    parser.add_argument("--test_dataset", type=str, default="swerick_data_random_test.pkl", help="pkl file of the test dataset")
    parser.add_argument("--valid_dataset", type=str, default="swerick_data_random_valid.pkl", help="pkl file of the valid dataset")
    parser.add_argument("--tokenizer_file", type=str, default="tokenizer_swerick.json", help="Save location for the tokenizer")
    args = parser.parse_args()

    main(args)