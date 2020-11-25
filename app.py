# Windows 10 used to write and test the code.
import os
import re
import pyprind
import numpy as np
import pandas as pd
from collections import Counter
from scripts.constants.sa_constants import base_path
from scripts.utils.sa_logging import get_logger

logger = get_logger()

# fixing random seed for reproducibility
np.random.seed(123)

stop_words = ['a', 'in', 'on', 'at', 'and', 'or',
              'to', 'the', 'of', 'an', 'by',
              'as', 'is', 'was', 'were', 'been', 'be',
              'are', 'for', 'this', 'that', 'these', 'those', 'you', 'i',
              'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'his', 'her', 'they', 'them', 'from', 'with', 'its']


def get_pre_process_data():
    labels = {'pos': 1, 'neg': 0}
    progress_bar = pyprind.ProgBar(50000)
    movie_reviews_df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(base_path, s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                movie_reviews_df = movie_reviews_df.append([[txt, labels[l]]], ignore_index=True)
                progress_bar.update()

    movie_reviews_df.columns = ["Review", "Sentiment"]

    movie_reviews_df = movie_reviews_df.reindex(np.random.permutation(movie_reviews_df.index))
    movie_reviews_df.to_csv('movie_data.csv', index=False, encoding='utf-8')
    movie_reviews_df = pd.read_csv('movie_data.csv', encoding='utf-8')

    pos_review_df = movie_reviews_df.loc[movie_reviews_df['Sentiment'] == 1]  # 25000
    neg_review_df = movie_reviews_df.loc[movie_reviews_df['Sentiment'] == 0]  # 25000

    # Split into 70%, 10% and 20% for training, validation and testing respectively
    pos_review_tr, pos_review_dev, pos_review_ts = np.split(pos_review_df.sample(frac=1, random_state=42),
                                                            [int(.7 * len(pos_review_df)),
                                                             int(.8 * len(pos_review_df))])

    neg_review_tr, neg_review_dev, neg_review_ts = np.split(neg_review_df.sample(frac=1, random_state=42),
                                                            [int(.7 * len(neg_review_df)),
                                                             int(.8 * len(neg_review_df))])

    # Create train, dev and test sets of both positive and negative reviews
    data_tr = pos_review_tr.append(neg_review_tr, ignore_index=True)
    data_dev = pos_review_dev.append(neg_review_dev, ignore_index=True)

    # Shuffle all the dataframes
    data_tr = data_tr.sample(frac=1).reset_index(drop=True)
    data_dev = data_dev.sample(frac=1).reset_index(drop=True)

    labels_tr = data_tr["Sentiment"].to_numpy()
    review_tr = data_tr["Review"].tolist()

    labels_dev = data_dev["Sentiment"].to_numpy()
    review_dev = data_dev["Review"].tolist()

    return labels_tr, review_tr, labels_dev, review_dev


def extract_ngrams(x_raw, ngram_range=(1, 3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', stop_words=[], vocab=set()):
    tuples = []

    # remove all non-words from the string
    words = re.compile(token_pattern)
    words_extract = words.findall(x_raw)

    # set all words to lower case (this was implemented due to the second task having somewhat unprocessed text)
    words_extract = [x.lower() for x in words_extract]

    # remove all words that are stop words
    words_extract = [n for n in words_extract if n not in stop_words]

    # construct ngrams between ngram_range[0] and ngram_range[1]
    for i in range(ngram_range[0], ngram_range[1] + 1):
        if i < 2:
            continue

        # find n-grams of length i
        for n in range(len(words_extract) + 1):

            # check theoretical tuple is valid ie n+i < len(x_raw)
            if n + i >= len(words_extract) + 1:
                break

            # take slice of n:n+i and turn into tuple, add tuple to array for later concatenation
            tuples.append(tuple(words_extract[n:n + i]))

    words_extract += tuples

    # remove all words not in vocab unless  vocab is default then ignore vocab
    if vocab != set():
        words_extract = [n for n in words_extract if n in vocab]

    return words_extract


def get_vocab(X_raw, ngram_range=(1, 3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', min_df=0, keep_topN=0, stop_words=[]):
    vocab = set()
    ngram_counts = Counter()  # term frequencies
    df_counter = Counter()  # document frequencies

    # loop through each element in the array X_raw
    for i in range(len(X_raw)):
        # extract the ngrams from the text
        extracted_words = extract_ngrams(X_raw[i], ngram_range, token_pattern, stop_words)

        # update the df counter (transform to set to ensure you only count one of each word per doc)
        df_counter.update(set(extracted_words))

        # update the tf counter
        ngram_counts.update(extracted_words)

    # ensure that topN_cutoff is the highest threshold
    if keep_topN < min_df:
        keep_topN = min_df

    # keep the top 'keep_topN' df's
    df_counter = Counter(dict(df_counter.most_common(keep_topN)))

    # keep the top 'keep_topN' tf's
    for word in list(ngram_counts):
        if word not in df_counter:
            del ngram_counts[word]

    # make a list of the vocabulary
    vocab, _ = map(set, zip(*df_counter.most_common(keep_topN)))

    return vocab, df_counter, ngram_counts


def corpus_to_ngram(X_raw, vocab, ngram_range=(1, 3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b'):
    # Define the return object
    X_ngram = []

    # turns a list of documents into a list of documents turned into ngrams

    # loop through each element in the list X_raw
    for i in range(len(X_raw)):
        # extract the ngrams from the text
        extracted_words = extract_ngrams(X_raw[i], ngram_range, token_pattern, [], vocab)

        # append the ngrams to the list
        X_ngram.append(extracted_words)

    return X_ngram


def vectorise(X_ngram, vocab, df=None):
    X_vec = np.zeros((len(X_ngram), len(vocab)))
    vocab = list(vocab)

    for i in range(len(X_ngram)):

        # a count fo each word in the document
        doc = Counter(X_ngram[i][:])

        for j in range(len(vocab)):

            if (vocab[j] in X_ngram[i][:]) and df != None:

                # Create tfidf count
                X_vec[i][j] = doc[vocab[j]] * np.log10(len(df) / df[vocab[j]])

            elif vocab[j] in X_ngram[i][:]:

                # Create a normal count matrix
                X_vec[i][j] = doc[vocab[j]]

    return X_vec


def sigmoid(z):
    # This fn includes a replacement for very low values to avoid overflow errors
    if isinstance(z, np.ndarray):
        z[z < -600] = -600
    elif z < -600:
        z = -600

    return 1 / (1 + np.exp(-z))


def predict_proba(X, weights):
    return sigmoid(np.dot(X, weights))


def predict_class(X, weights):
    preds_class = predict_proba(X, weights)

    preds_class[preds_class >= 0.5] = 1
    preds_class[preds_class < 0.5] = 0

    return preds_class


def binary_loss(X, Y, weights, alpha=0.00001):
    # using X and weights make an array of probability predictions
    Y_prob = predict_proba(X, weights)

    # regularisation
    reg = np.sum(np.square(weights))

    # compare predictions to Y and calculate L_vec for every Y
    l_vec = -Y * np.log(Y_prob) - (1 - Y) * np.log(1 - Y_prob) + alpha * reg

    return np.sum(l_vec) / len(Y)


def SGD(X_tr, Y_tr, X_dev=[], Y_dev=[], loss="binary", lr=0.1, alpha=0.00001, epochs=5, tolerance=0.0001, print_progress=True):
    tr_loss_history = []
    val_loss_history = []

    # Initalise weights
    w = np.zeros(X_tr.shape[1])

    for i in range(epochs):

        # randomise the order in the data every epoch
        randomise = np.arange(len(Y_tr))
        np.random.shuffle(randomise)
        X_tr = X_tr[randomise]
        Y_tr = Y_tr[randomise]

        # Over the dataset modify the weights
        for j in range(len(Y_tr)):
            grad = (predict_proba(w, X_tr[j, :]) - Y_tr[j]) * X_tr[j, :] + 2 * alpha * w
            w -= lr * grad

        # Calculate the training and validation loss after each epoch
        tr_loss_history.append(binary_loss(X_tr, Y_tr, w, alpha))
        val_loss_history.append(binary_loss(X_dev, Y_dev, w, alpha))

        if print_progress:
            print("Epoch: %3i|\t Training loss: %7f|\t Validation loss: %7f"
                  % (i, tr_loss_history[i], val_loss_history[i]))

        # test to see if validation set is getting larger or the change is smaller than tolerance
        if (i > 0) and (i != epochs - 1) and ((val_loss_history[i - 1] - val_loss_history[i]) < tolerance):
            break

    return w, tr_loss_history, val_loss_history


def main(review_ts):
    logger.info("STARTING SENTIMENT ANALYSIS APPLICATION")

    labels_tr, review_tr, labels_dev, review_dev = get_pre_process_data()
    vocab, df, ngram_counts = get_vocab(review_tr, ngram_range=(1, 3), keep_topN=5000, stop_words=stop_words)

    logger.info("GOT DATA")

    tr_ngram = corpus_to_ngram(review_tr, vocab)
    ts_ngram = corpus_to_ngram(review_ts, vocab)
    dev_ngram = corpus_to_ngram(review_dev, vocab)

    logger.info("CREATED NGRAMS")

    tr_vec = vectorise(tr_ngram, vocab)
    ts_vec = vectorise(ts_ngram, vocab)
    dev_vec = vectorise(dev_ngram, vocab)

    logger.info("VECTORIZED NGRAMS")

    w_count, loss_tr_count, dev_loss_count = SGD(tr_vec, labels_tr, dev_vec, labels_dev, lr=0.0001, alpha=0.001,
                                                 epochs=200)

    logger.info("RAN SGD MODEL")

    label_predict = predict_class(ts_vec, w_count)

    logger.info("PROCESS IS COMPLETE")
    return label_predict


if __name__ == '__main__':
    main(review_ts)
