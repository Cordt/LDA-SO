from gensim import *

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import math
import random
import uuid


class Tagcloud(object):

    FONT = '/System/Library/Fonts/Avenir.ttc'
    FONT_COLOR = ['#F2B701', '#E57D04', '#DC0030', '#B10058', '#7C378A', '#3465AA', '#09A275', '#85BC5F',
                  '#39d', '#aab5f0']
    FONT_SIZE = [15, 18, 20, 22, 24, 27, 30, 35, 40, 45]
    
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.words = list
        self.words_to_draw = None
        self.image = PIL.Image.new('RGBA', [width, height], "#fff")
        self.imageDraw = PIL.ImageDraw.Draw(self.image)
        self.imagefilepath = str

    @staticmethod
    def formatwordlist(wordlist, values):
        if not isinstance(wordlist, list):
            raise ValueError('words should be a list')

        if not isinstance(values, list):
            raise ValueError('values should be a list')

        formattedwords = []
        count = 0
        for aword in wordlist:
            formattedwords.append({"text": aword, "weight": values[count]})
            count += 1

        return formattedwords

    def draw(self, wordlist, imagefilepath=None):
        self.words = wordlist
        if imagefilepath is None:
            imagefilepath = str(uuid.uuid4()) + '.jpg'
        self.imagefilepath = imagefilepath

        index = 0
        length = len(self.words)
        for aword in self.words:
            if index == length - 1:
                weight = 0
            else:
                weight = self._rescaleweight(aword['weight'], self.words[0]['weight'], self.words[-1]['weight'])
            self._findcoordinates(index, aword['text'], int(weight))
            index += 1

        return self._save()

    @staticmethod
    def _rescaleweight(n, maxinum, minimum):
        scalemin = 1
        scalemax = 10

        # if max and min is same return max weight - 1
        if maxinum == minimum:
            return 9

        weight = round((1.0 * (scalemax - scalemin) * (n - minimum)) / (maxinum-minimum))

        return weight

    def _findcoordinates(self, index, text, weight):
        anglestep = 0.57
        radiusstep = 8
        radius = 25
        angle = random.uniform(0.2, 6.28)

        fontsize = self.FONT_SIZE[weight]
        width, height = self.imageDraw.textsize(text, font=PIL.ImageFont.truetype(self.FONT, fontsize))

        x = self.width/2 - width/2.0
        y = self.height/2 - height/2.0

        count = 1
        while self._checkoverlap(x, y, height, width):
            if count % 8 == 0:
                radius += radiusstep
            count += 1

            if index % 2 == 0:
                angle += anglestep
            else:
                angle += -anglestep

            x = self.width/2 - (width / 2.0) + (radius*math.cos(angle))
            y = self.height/2 + radius*math.sin(angle) - (height / 2.0)

        self.words_to_draw.append({'text': text, 'fontsize': fontsize, 'x': x, 'y': y, 'w': width,
                                   'h': height, 'color': self.FONT_COLOR[weight]})

    def _checkoverlap(self, x, y, h, w):
        if not self.words_to_draw:
            self.words_to_draw = []
            return False

        for aword in self.words_to_draw:
            if not ((x+w < aword['x']) or (aword['x'] + aword['w'] < x) or (y + h < aword['y'])
                    or (aword['y'] + aword['h'] < y)):
                return True

        return False

    def _save(self):
        for aword in self.words_to_draw:
            if self._liesinside(aword):
                self.imageDraw.text((aword['x'], aword['y']), aword['text'],
                                    font=PIL.ImageFont.truetype(self.FONT, aword['fontsize']), fill=aword['color'])

        self.image.save(self.imagefilepath, "JPEG", quality=90)
        return self.imagefilepath

    def _liesinside(self, aword):
        if aword['x'] >= 0 and aword['x'] + aword['w'] <= self.width \
                and aword['y'] >= 0 and aword['y'] + aword['h'] <= self.height:
            return True

        return False


if __name__ == '__main__':

    # dictionary = corpora.Dictionary.load("topic_model_lda/dict.dict")
    # tfidf_model = models.TfidfModel.load("topic_model_lda/lda_model_tfidf_model.model")
    # lda_model = models.LdaModel.load("topic_model_lda/lda_model.model")
    #
    # for topic in range(0, lda_model.num_topics):
    #     words = []
    #     t = TagCloud()
    #     topic_words = lda_model.show_topic(topic, 100)
    #     for (prob, word) in topic_words:
    #         words.append({"text": word, "weight": prob})
    #     print t.draw(words)

    t = Tagcloud()
    words = [{"text": "coffee", "weight": 20296.0}, {"text": "love", "weight": 15320.0},
             {"text": "day", "weight": 6860.0}, {"text": "like", "weight": 5521.0},
             {"text": "follow", "weight": 5393.0}, {"text": "morning", "weight": 5125.0},
             {"text": "happy", "weight": 5099.0}, {"text": "girl", "weight": 5049.0},
             {"text": "cute", "weight": 4336.0}, {"text": "good", "weight": 4328.0},
             {"text": "tumblr", "weight": 4169.0}, {"text": "today", "weight": 4142.0},
             {"text": "followme", "weight": 3923.0}, {"text": "chocolate", "weight": 3922.0},
             {"text": "instagood", "weight": 3818.0}, {"text": "yummy", "weight": 3786.0},
             {"text": "new", "weight": 3700.0}, {"text": "lol", "weight": 3536.0},
             {"text": "yum", "weight": 3282.0}, {"text": "drink", "weight": 3246.0},
             {"text": "latte", "weight": 3219.0}, {"text": "time", "weight": 3137.0},
             {"text": "caramel", "weight": 3125.0}, {"text": "friends", "weight": 3084.0},
             {"text": "tagsforlikes", "weight": 3005.0}, {"text": "beautiful", "weight": 2892.0},
             {"text": "food", "weight": 2801.0}, {"text": "life", "weight": 2780.0},
             {"text": "delicious", "weight": 2767.0}, {"text": "f4f", "weight": 2670.0},
             {"text": "follow4follow", "weight": 2667.0}, {"text": "white", "weight": 2614.0},
             {"text": "tea", "weight": 2577.0}, {"text": "selfie", "weight": 2559.0},
             {"text": "best", "weight": 2522.0}, {"text": "swag", "weight": 2494.0},
             {"text": "got", "weight": 2491.0}, {"text": "work", "weight": 2479.0},
             {"text": "fashion", "weight": 2458.0}, {"text": "likeforlike", "weight": 2440.0},
             {"text": "amazing", "weight": 2434.0}, {"text": "followforfollow", "weight": 2366.0},
             {"text": "get", "weight": 2365.0}, {"text": "fun", "weight": 2273.0},
             {"text": "like4like", "weight": 2236.0}, {"text": "frappuccino", "weight": 2167.0},
             {"text": "picoftheday", "weight": 2083.0}, {"text": "breakfast", "weight": 2062.0},
             {"text": "smile", "weight": 2060.0}, {"text": "photooftheday", "weight": 2016.0},
             {"text": "summer", "weight": 1982.0}, {"text": "hot", "weight": 1928.0},
             {"text": "mocha", "weight": 1906.0}, {"text": "instadaily", "weight": 1896.0},
             {"text": "pink", "weight": 1883.0}, {"text": "perfect", "weight": 1804.0},
             {"text": "shopping", "weight": 1801.0}]
    print t.draw(words)