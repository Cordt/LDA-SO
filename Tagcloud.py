import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import math
import random
import uuid
import os


class Tagcloud(object):

    FONT = '/System/Library/Fonts/Avenir.ttc'
    FONT_COLOR = ['#F2B701', '#E57D04', '#DC0030', '#B10058', '#7C378A', '#3465AA', '#09A275', '#85BC5F',
                  '#39d', '#aab5f0']
    FONT_SIZE = [15, 18, 20, 22, 24, 27, 30, 35, 40, 45]
    
    def __init__(self, setting, width=400, height=400):
        self.setting = setting
        self.width = width
        self.height = height
        self.words = list
        self.words_to_draw = None
        self.image = PIL.Image.new('RGBA', [width, height], "#fff")
        self.imagedraw = PIL.ImageDraw.Draw(self.image)
        self.imagefilepath = str

    @staticmethod
    def createtagcloud(topicmodel, setting):
        logging.info("Drawing tagcloud...")
        count = 0
        directory = setting['resultfolder'] + setting['theme'] + "/tagclouds/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for topic in topicmodel.model.show_topics(num_topics=-1, num_words=setting['noofwordsfortopic'],
                                                  formatted=False):
            tagcloud = Tagcloud(setting)
            words = []
            for (prob, word) in topic:
                words.append({"text": word, "weight": prob})

            filename = "Topic-" + str(count) + ".jpg"
            path = ''.join([directory, filename])
            tagcloud._draw(words, imagefilepath=path)
            count += 1

    @staticmethod
    def _formatwordlist(wordlist, values):
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

    def _draw(self, wordlist, imagefilepath=None):
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
        width, height = self.imagedraw.textsize(text, font=PIL.ImageFont.truetype(self.FONT, fontsize))

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
                self.imagedraw.text((aword['x'], aword['y']), aword['text'],
                                    font=PIL.ImageFont.truetype(self.FONT, aword['fontsize']), fill=aword['color'])

        self.image.save(self.imagefilepath, "JPEG", quality=90, replace=True)
        return self.imagefilepath

    def _liesinside(self, aword):
        if aword['x'] >= 0 and aword['x'] + aword['w'] <= self.width \
                and aword['y'] >= 0 and aword['y'] + aword['h'] <= self.height:
            return True

        return False
