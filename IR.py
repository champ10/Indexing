from __future__ import division
import glob
import os, sys
import numpy as np
import codecs
import nltk
from collections import defaultdict, OrderedDict
from nltk.stem import WordNetLemmatizer
import pickle
import logging
from datetime import datetime
import time

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info('script started at %s' % (str(datetime.now())))
script_start_time = time.time()


def getfilenames(base_dir):
    '''
    retrieve file names from dir

    :param base_dir:
    :return:
    '''
    filenames = glob.glob(base_dir + '/*.txt')
    base_filenames = [os.path.basename(a_path) for a_path in filenames]
    base_filenames.remove(index_file + ".txt")
    doc_ids = range(len(base_filenames))
    docdict = dict(zip(doc_ids, base_filenames))
    return docdict


def getContent(filename, encoding='latin1'):
    '''
    get text content from file

    :param filename:
    :param encoding:
    :return:
    '''
    with codecs.open(filename, mode='r', encoding=encoding) as inf:
        text = inf.read()
    return text


class Normaliser(object):
    '''
    Text preprocessing class. Responsible for term normalization using stop word and lemmatization
    '''

    def __init__(self, lemmatizer=None, stop_words=None, dictionary=None, lower_case=True):
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
        print("The number of stop words is %d" % (len(self.stop_words)))
        print("lemmatizer is " + str(self.lemmatizer))
        self.dictionary = dictionary
        self.lower_case = lower_case

    def normalise(self, token):
        '''
        normalization  using stop word removal and lemmatization
        :param token:
        :return:
        '''
        if self.lower_case:
            token = token.lower()
        if token in self.stop_words:
            return None
        if self.lemmatizer:
            token = self.lemmatizer.lemmatize(token)
        if self.dictionary:
            if token not in self.dictionary:
                return None
        return token


class Indexer(object):
    '''
    Indexer class responsibel for generating and storing inverted indexes
    '''

    def __init__(self, tokeniser, normaliser=None):
        self.tokeniser = tokeniser
        self.normaliser = normaliser
        self.inverted_index = defaultdict(PostingList)
        # total number of documents
        self.N = 0
        self.document_lengths = defaultdict(float)
        self.dl = defaultdict(int)
        self.dic = defaultdict(int)

    def index(self, docID, text):
        '''
        Generate index from document

        :param docID:
        :param text:
        :return:
        '''
        tokens = self.tokeniser.tokenize(text)
        token_position = 0
        term_documents = defaultdict(TermDocument)
        for token in tokens:
            if self.normaliser:
                token = self.normaliser.normalise(token)
            if not token:
                continue
            term_document = term_documents[token]
            term_document.tf += 1
            term_document.positions.append(token_position)
            token_position += 1
            self.dl[docID] += 1
            self.dic[token] += 1
        # update the main index
        for term, term_document in term_documents.items():
            tf = term_document.tf
            self.document_lengths[docID] += np.square(tf)
            self.inverted_index[term].posts.append([docID, term_document])
            self.inverted_index[term].df += 1
        self.N += 1
        self.document_lengths[docID] = np.sqrt(self.document_lengths[docID])

    def dump(self, filename):
        '''
        Storing raw indexes and pickel file

        :param filename:
        :return:
        '''
        logging.info("dumping index to %s" % (filename))
        with open(filename, 'wb') as outf:
            pickle.dump((self.inverted_index, self.document_lengths, self.N, self.document_lengths, self.dl, self.dic),
                        outf)
        with open(filename + ".txt", 'w') as outf:
            for item in self.inverted_index.items():
                term = item[0]
                text = str(term)
                for post in item[1].posts:
                    doc_id = post[0]
                    tf = post[1].tf
                    text = text + ":" + "(doc:" + str(doc_id) + ",tf:" + str(tf) + "), "
                # print(text)
                outf.write(text)

    def load(self, filename):
        '''
        Loads  indexes from file

        :param filename:
        :return:
        '''
        logging.info("loading index from %s" % (filename))
        with open(filename, 'rb') as inf:
            self.inverted_index, self.document_lengths, self.N, self.document_lengths, self.dl, self.dic = pickle.load(
                inf)

    def search(self, query):
        '''
        Searching query from indxes

        :param query:
        :return:
        '''
        results = defaultdict(float)
        for term in query:
            posting_list = self.inverted_index[term]
            df = posting_list.df
            idf = np.log(self.N / (df + 1))
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                tfidf = tf * idf
                results[docID] += tfidf
        for docID in results:
            results[docID] = results[docID] / self.document_lengths[docID]
        ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked_results


class QueryProcessor(object):
    '''
    Responsible for query preprocessing
    '''

    def __init__(self, tokeniser, normaliser=None):
        self.tokeniser = tokeniser
        self.normaliser = normaliser

    def process(self, query):
        tokens = self.tokeniser.tokenize(query)
        query_terms = []
        for token in tokens:
            if self.normaliser:
                token = self.normaliser.normalise(token)
            if not token:
                continue
            query_terms.append(token)
        query_terms = self._expandQuery(query_terms)
        return query_terms

    def _expandQuery(self, query_terms):
        return query_terms


class PostingList(object):
    def __init__(self):
        # each post is like [docid, term_document] and a term_document is tf + positions
        self.posts = []
        # the number of documents a term occurred in.
        self.df = 0


class TermDocument(object):
    def __init__(self):
        # the number of times a term occurs in a document
        self.tf = 0
        self.positions = []


class NLTKWordTokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.word_tokenize(text)


def load_queries(query_file):
    logging.info("loading queries...")
    queries = {}
    with codecs.open(query_file, mode='r', encoding=encoding) as inf:
        for line in inf:
            fields = line.split('\t')
            queryID = int(fields[0])
            query = fields[1]
            queries[queryID] = query
    return queries


def get_docname_id(docs):
    docname_id = {}
    for id, docName in docs.iteritems():
        docname_id[docName] = id
    return docname_id


def indexing(base_dir, indexer):
    docs = getfilenames(base_dir=base_dir)
    docs_length = len(docs)
    logging.info("Indexing %d docs in %s" % (docs_length, base_dir))
    docs_processed = 1
    for docID, filename in docs.items():
        # if docs_processed % tenpercent == 0:
        #     logging.info("processing " + str(10 * docs_processed / tenpercent) + "%")
        docs_processed += 1
        filename = os.path.join(base_dir, filename)
        text = getContent(filename, encoding=encoding)
        indexer.index(docID, text)
    indexer.dump(base_dir + index_file)


def searching(base_dir, indexer, query_processor, query):
    '''
    Search given string query in index
    :param base_dir:
    :param indexer: indexer obj
    :param query_processor: query processor obj
    :param query: string query
    :return:
    '''
    # print(query)
    docs = getfilenames(base_dir=base_dir)
    docs_length = len(docs)

    op = np.array([False] * docs_length)
    query_terms = query_processor.process(query)
    results = indexer.search(query_terms)
    for result in results:
        docID, score = result
        op[docID] = True
        # docName = docs[docID]
        # print("DocID: %d DocName: %s Score: %0.2f" % (docID, docName, score))
    return op


def serach_query(base_dir, indexer, query_processor):
    '''
    seraches query in given index

    :param base_dir:  directory path for stored index
    :param indexer: indexer obj
    :param query_processor: query processor obj
    :return:
    '''
    docs = getfilenames(base_dir=base_dir)
    indexer.load(base_dir + index_file)
    while (True):
        query = input("Please enter query:")
        if query == "q:":
            break
        query = query.replace("(", " ( ")
        query = query.replace(")", " ) ")
        str_eval = ""
        for split_word in query.split():
            if split_word == "(" or split_word == ")":
                str_eval += split_word + " "
            elif split_word == "OR":
                str_eval += "| "
            elif split_word == "AND":
                str_eval += "& "
            elif split_word == "NOT":
                str_eval += "~ "
            else:
                str_eval += "searching(base_dir,indexer,query_processor, '" + split_word + "')" + " "
        # print(str_eval)

        op = eval(str_eval)
        for docID in np.where(np.array(op) == True)[0]:
            docName = docs[docID]
            print("DocName: %s" % docName)


index_file = 'index.pkl'
encoding = 'latin1'

'''
 main function with 2 command line argument
 1. i for indexing or s for serach query
 2. directory path for input data or stored index
'''
if __name__ == "__main__":

    program_name = sys.argv[0]
    arguments = sys.argv[1:]
    count = len(arguments)
    if count == 2:
        # initializing lemaatizer, stopword , indexer and query processor
        lemmatizer = WordNetLemmatizer()

        normaliser = Normaliser(lemmatizer=lemmatizer, stop_words=nltk.corpus.stopwords.words('english'),
                                dictionary=None,
                                lower_case=True)
        tokeniser = NLTKWordTokenizer()
        indexer = Indexer(tokeniser, normaliser)
        query_processor = QueryProcessor(tokeniser, normaliser)

        base_dir = arguments[1]

        if arguments[0].lower() == "i":
            # generate  index
            indexing(base_dir, indexer)
        elif arguments[0].lower() == "s":
            # retrieve document using query
            serach_query(base_dir, indexer, query_processor)

    else:
        print("invalid command line argument...")