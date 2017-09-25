import numpy as np
import urllib.request as http


# Counts number of word occurrences
# Param:  List<Words>
# Return: List<Dict<Word, Count>, TotalWords>
def count_words(_words):
    word_counts = dict()
    words = _words.split()
    total_words = 0

    for word in words:

        word = word.lower()

        if word.isalnum():

            if word in word_counts:
                word_counts[word] += 1
                total_words += 1
            else:
                word_counts[word] = 1
                total_words += 1

    return word_counts, total_words


# Reads text file into list
# Param:  File name
# Return: List
def load_text_file(file):

    with open(file) as f:
        text_file = f.read().splitlines()

    return text_file


# Calculates Term Frequency
# Param:  Dict<Word, Count>, TotalWords
# Return: Dict<Word, TF>
def find_tf(_word_counts, _total_words):

    term_freq = {}
    i = 0

    for word, count in _word_counts.items():

        tf = (count / _total_words)
        term_freq[i] = word, tf
        i += 1

    return term_freq


# HTTP Get Request
# Param:  URL  String
# Return: HTML String
def get_src(_url):
    resp = http.urlopen(_url)
    _src = resp.read()
    return _src


# Calculates Inverse Document Frequency
# Param: Dict<Word, "">, List[Dict<Word, Count>]
# Return: Dict<Word, IDF_Score>
def find_idf(_words, _docs):

    idf_scores = dict()
    num_docs = _docs.__len__()
    num_docs_with_word = 0

    for word in _words:

        for doc in _docs:

            if word in doc:
                num_docs_with_word += 1

        idf = 1 + np.log(num_docs/num_docs_with_word)
        idf_scores[word] = idf
    return idf_scores


# TODO: Check both dicts have all the words
# Calculates Term Frequency - Inverse Document Frequency TF-IDF
# Param: Dict<Word, TF_IDF>
# Return: Dict<>
def find_tf_idf(_word_tf_scores, _word_idf_scores, _word_tf_idf):

    tf_idf_scores = {}

    for _word, _tf_score in _word_tf_scores.items():
        word = _tf_score[0]
        tf_score = _tf_score[1]

        if word not in word_tf_idf:
            tf_idf_scores[word] = tf_score * _word_idf_scores[word]

    return tf_idf_scores

urls = load_text_file("urls")
i = 0

# List Docs<Word, Count>
docs_word_counts = []

# List Docs<Word, TF Score>
word_tf = []

# Dict of all words from all documents
word_dict = {}

#
word_tf_idf = {}

for url in urls:
    # Get page source
    src = get_src(url)

    # Doc<Word, WordCount>   = word_counts
    # Total Number of Words  = total_words
    word_counts, total_words = count_words(src)

    # Add Doc<Word, Count> to a List
    docs_word_counts.append(word_counts)

    # Find TF
    # Add Doc<Word, TF> to a List
    # TODO: CALL THIS ONCE?
    word_tf.append(find_tf(docs_word_counts[i], total_words))

    # Combine dicts into one dict with all words
    # TODO: Get all the words in a list instead!?
    word_dict = {**word_dict, **docs_word_counts[i]}
    i += 1


word_idf = find_idf(word_dict, docs_word_counts)

for doc in word_tf:
    word_tf_idf = find_tf_idf(word_tf[0], word_idf, word_tf_idf)


# Step 1: (Term Frequency)             - (Word[0].Count / Total document words)
# Step 2: (Inverse Document Frequency) - 1 + loge(Num Docs      / Num Docs with term 'word[0]' )
# Step 3: (TF * IDF)                   - https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
print()



