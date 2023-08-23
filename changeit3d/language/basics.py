import numpy as np
import pandas as pd
from functools import partial
from collections import defaultdict
from symspellpy.symspellpy import SymSpell

from .language_preprocessing import unquote_words, expand_contractions
from .language_preprocessing import manual_sentence_spelling, manual_tokenized_sentence_spelling
from .spelling import sentence_spelling_dictionary, missing_from_glove_but_are_actual_words
from .comparative_superlative import break_down_superlative_comparatives
from ..utils.basics import parallel_apply


def load_glove_pretrained_embedding(glove_file, dtype=np.float32, only_words=False, verbose=False):
    """
    :param glove_file: file downloaded from Glove website
    :param dtype: how to save the word-embeddings
    :param only_words: do not return the embedding vectors, only the words considered
    :param verbose: print, or not side-information
    :return: dictionary of words mapped to np.array vectors
    """

    if verbose:
        print("Loading glove word embeddings.")

    embedding = dict()
    with open(glove_file) as f_in:
        for line in f_in:
            s_line = line.split()
            token = s_line[0]
            if only_words:
                embedding[token] = 0
            else:
                w_embedding = np.array([float(val) for val in s_line[1:]], dtype=dtype)
                embedding[token] = w_embedding
    if only_words:
        embedding = set(list(embedding.keys()))

    if verbose:
        print("Done.", len(embedding), "words loaded.")
    return embedding


def tokenize_and_spell(df, glove_file, freq_file, tokenizer, inplace=True, spell_check=True, token_spelling_dictionary=None):
    speller = SymSpell()
    loaded = speller.load_dictionary(freq_file, term_index=0, count_index=1)
    print('SymSpell spell-checker loaded:', loaded)
    golden_vocabulary = load_glove_pretrained_embedding(glove_file, only_words=True, verbose=True)
    golden_vocabulary = golden_vocabulary.union(missing_from_glove_but_are_actual_words)
    print('Updating Glove vocabulary with *valid* ShapeTalk words that are missing from it.')
    missed_tokens = defaultdict(list)

    def automatic_token_speller(token_list, max_edit_distance=1):
        new_tokens = []
        for token in token_list:
            if token in golden_vocabulary:
                new_tokens.append(token)  # no spell check
            else:
                spells = speller.lookup(token, max_edit_distance)
                if len(spells) > 0:  # found a spelled checked version
                    new_tokens.append(spells[0].term)
                else:  # spell checking failed
                    context = " ".join(token_list)
                    missed_tokens[token].append(context)
                    new_tokens.append(token)
        return new_tokens

    if not spell_check:
        automatic_token_speller = None

    spelled_tokens = pre_process_text(df.utterance,
                                      sentence_spelling_dictionary,
                                      tokenizer,
                                      manual_token_speller=token_spelling_dictionary,
                                      automatic_token_speller=automatic_token_speller)

    all_conversions = set()
    spelled_tokens = spelled_tokens.apply(partial(break_down_superlative_comparatives,
                                                  all_conversions=all_conversions))
    # print(len(all_conversions), all_conversions)

    if inplace:
        df['tokens'] = spelled_tokens
        df['tokens_len'] = df.tokens.apply(lambda x: len(x))
        df['utterance_spelled'] = df.tokens.apply(lambda x: ' '.join(x))
        return missed_tokens
    else:
        return missed_tokens, spelled_tokens


def pre_process_text(text, manual_sentence_speller, tokenizer,  manual_token_speller=None, automatic_token_speller=None):

    # replace verbatim entire sentences
    clean_text = text.apply(lambda x: manual_sentence_spelling(x, manual_sentence_speller))  # sentence-to-sentence map

    # lowercase
    clean_text = clean_text.apply(lambda x: x.lower())

    # unquote
    clean_text = clean_text.apply(unquote_words)

    # expand_contractions
    clean_text = pd.Series(parallel_apply(clean_text, expand_contractions))

    # map " and ` and ’ to '
    clean_text = clean_text.apply(lambda x: x.replace('\"', '\''))
    clean_text = clean_text.apply(lambda x: x.replace("\`", '\''))
    clean_text = clean_text.apply(lambda x: x.replace('\’', '\''))

    pedantic = True
    if pedantic:
        # replace some common expressions
        clean_text = clean_text.apply(lambda x: x.replace(' w/ ', ' with '))
        clean_text = clean_text.apply(lambda x: x.replace('&', ' and '))
        clean_text = clean_text.apply(lambda x: x.replace('it ’ s', ' it is '))

    # map punctuation and some symbols to space
    basic_punct_and_symbs = '.?!,:;/\-~*_=[–]{}$^@|%#<>()'
    punct_to_space = str.maketrans(basic_punct_and_symbs, ' ' * len(basic_punct_and_symbs))  # map punctuation to space
    clean_text = clean_text.apply(lambda x: x.translate(punct_to_space))

    # tokenize
    clean_tokens = pd.Series(parallel_apply(clean_text, tokenizer))

    # apply manual-spell check at token-level
    if manual_token_speller is not None:
        clean_tokens = clean_tokens.apply(lambda x:
                                          manual_tokenized_sentence_spelling(x, spelling_dictionary=manual_token_speller))

    if automatic_token_speller is not None:
        clean_tokens = clean_tokens.apply(automatic_token_speller)

    return clean_tokens




