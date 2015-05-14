__author__ = 'Cordt'

from gensim import corpora, models, utils


class Topicmodel:

    def __init__(self, preprodata, setting):
        mallet_path = '/usr/share/mallet-2.0.7/bin/mallet'
        self.corpus = preprodata.corpus
        self.vocabulary = preprodata.vocabulary
        nooftopics = setting['nooftopics']
        noofiterations = setting['noofiterations']
        self.model = models.wrappers.LdaMallet(mallet_path, self, num_topics=nooftopics, id2word=self.vocabulary,
                                               iterations=noofiterations)

    def __iter__(self):
        for tokens in self.corpus:
            yield self.vocabulary.doc2bow(tokens)

    # def __init__(self, vocabulary, doctermmatrix):
    #     self.doctermmatrix = doctermmatrix
    #     self.vocabulary = vocabulary
    #     print("Defining topic model...")
    #     self.model = lda.LDA(n_topics=20, n_iter=800, random_state=1)
    #
    # def definemodel(self):
    #     print("Fitting topic model...")
    #     self.model.fit(self.doctermmatrix)
    #     topicword = self.model.topic_word_
    #     ntopwords = 16
    #     for i, topicdist in enumerate(topicword):
    #         topicwords = np.array(self.vocabulary)[np.argsort(topicdist)][:-ntopwords:-1]
    #         print('Topic {}: {}'.format(i, ' '.join(topicwords)))
    #
    # def toptopicsfortitles(self, titles):
    #     doc_topic = self.model.doc_topic_
    #     for i in range(len(titles)):
    #         print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))