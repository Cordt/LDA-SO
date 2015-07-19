__author__ = 'Cordt Voigt'

import xml.etree.cElementTree as ElementTree
import sqlite3
import os.path
import re
import logging


class Importer:

    def __init__(self, setting):
        # Instance variables
        self.setting = setting
        self.corpus = []
        self.connection = None
        self.cursor = None

    def import_xml_data(self):
        self._create_db()
        self._import_tags()
        self._import_posts()
        self._import_links()
        self._import_users()

    def get_number_of_questions(self):
        # Database connection - instance variables
        self.connection = sqlite3.connect(self.setting['dbpath'])
        self.cursor = self.connection.cursor()

        self.corpus = []

        sql = 'SELECT COUNT(id) FROM question'
        self.cursor.execute(sql)
        questions = self.cursor.fetchone()
        return questions[0]

    def get_question_corpus(self):
        # Database connection - instance variables
        self.connection = sqlite3.connect(self.setting['dbpath'])
        self.cursor = self.connection.cursor()

        self.corpus = []

        logging.info("Loading questions...")
        sql = 'SELECT title, body FROM question'
        self.cursor.execute(sql)
        questions = self.cursor.fetchall()
        for row in questions:
            # Append document to corpus
            self.corpus.append(' '.join([row[0], row[1]]))

        self.connection.close()
        return self.corpus

    def get_answer_corpus(self):
        # Database connection - instance variables
        self.connection = sqlite3.connect(self.setting['dbpath'])
        self.cursor = self.connection.cursor()

        self.corpus = []

        logging.info("Loading answers...")
        sql = 'SELECT body FROM answer'
        self.cursor.execute(sql)
        answers = self.cursor.fetchall()
        for row in answers:
            # Append document to corpus
            self.corpus.append(row[0])

        self.connection.close()
        return self.corpus

    def _create_db(self):
        # Create tables if database does not exist yet
        if not os.path.isfile(self.setting['dbpath']):
            logging.info("Database does not exist yet, creating...")

            # Database connection - instance variables
            self.connection = sqlite3.connect(self.setting['dbpath'])
            self.cursor = self.connection.cursor()

            # Create question table
            sql = 'CREATE TABLE question (id INTEGER PRIMARY KEY AUTOINCREMENT, elementId int, creationDate int, ' \
                  'score int, body text, ownerUserId int, lastActivityDate int, commentCount int, ' \
                  'acceptedAnswerId int, viewCount int, title text, answerCount int)'
            self.cursor.execute(sql)

            # Create answer table
            sql = 'CREATE TABLE answer (id INTEGER PRIMARY KEY AUTOINCREMENT, elementId int, creationDate int, ' \
                  'score int, body text, ownerUserId int, lastActivityDate int, commentCount int, ' \
                  'questionId int, lastEditorUserId int, lastEditDate int)'
            self.cursor.execute(sql)

            # Create postLink table
            # LinkTypeId:
            # 1: Mutual reference
            # 2: ??
            # 3: Marked as duplicate, referencing to duplicate question
            sql = 'CREATE TABLE postLink (id INTEGER PRIMARY KEY AUTOINCREMENT, elementId int, creationDate int, ' \
                  'postId int, relatedPostId int, linkTypeId int)'
            self.cursor.execute(sql)

            # Create tag table
            sql = 'CREATE TABLE tag (id INTEGER PRIMARY KEY AUTOINCREMENT, elementId int, tagName text, count int)'
            self.cursor.execute(sql)

            # Create postTag table
            sql = 'CREATE TABLE postTag (id INTEGER PRIMARY KEY AUTOINCREMENT, questionId int, tagId int)'
            self.cursor.execute(sql)

            # Create user table
            sql = 'CREATE TABLE user (id INTEGER PRIMARY KEY AUTOINCREMENT, elementId int, reputation int, ' \
                  'creationDate int, displayName text, lastAccessDate int, websiteUrl text, location text, ' \
                  'aboutMe text, views int, upVotes int, downVotes int, profileImageUrl text, age int, accountId int)'
            self.cursor.execute(sql)

            self.connection.commit()

        else:
            # Database connection - instance variables
            self.connection = sqlite3.connect(self.setting['dbpath'])
            self.cursor = self.connection.cursor()

            logging.info("Database already exists, connecting...")

    def _import_posts(self):
        tree = ElementTree.parse(self.setting['folderprefix'] + 'Posts.xml')
        root = tree.getroot()

        # Check whether database is empty
        sql = 'SELECT * FROM question'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        sql = 'SELECT id, questionId FROM answer'
        self.cursor.execute(sql)
        result2 = self.cursor.fetchall()

        if len(result) != 0 or len(result2) != 0:
            logging.info('Question and Answer tables not empty, adding nothing')

        else:
            logging.info('Importing posts into question and answer tables')
            for row in root.findall('row'):

                # Common attributes
                elementid = row.get('Id')
                creationdate = row.get('CreationDate')
                score = row.get('Score')
                body = row.get('Body')
                owneruserid = row.get('OwnerUserId')
                lastactivitydate = row.get('LastActivityDate')
                commentcount = row.get('CommentCount')
                post_type_id = row.get('PostTypeId')

                if post_type_id == '1':
                    acceptedanswerid = row.get('AcceptedAnswerId')
                    viewcount = row.get('ViewCount')
                    title = row.get('Title')
                    tags = row.get('Tags')
                    # Add tags for question
                    self._insert_tags_for_post(tags, elementid)
                    answercount = row.get('AnswerCount')
                    values = [elementid, creationdate, score, body, owneruserid, lastactivitydate, commentcount,
                              acceptedanswerid, viewcount, title, answercount]

                    self.cursor.execute('INSERT INTO question VALUES (NULL, ?, strftime(\'%s\', ?), ?, ?, ?, \
                                         strftime(\'%s\', ?), ?, ?, ?, ?, ?)', values)

                elif post_type_id == '2':
                    question_element_id = row.get('ParentId')
                    lasteditoruserid = row.get('LastEditorUserId')
                    lasteditdate = row.get('LastEditDate')
                    values = [elementid, creationdate, score, body, owneruserid, lastactivitydate, commentcount,
                              question_element_id, lasteditoruserid, lasteditdate]

                    self.cursor.execute('INSERT INTO answer VALUES (NULL, ?, strftime(\'%s\', ?), ?, ?, ?, \
                                         strftime(\'%s\', ?), ?, ?, ?, ?)', values)

            # Retrieve question ID (not question element ID) - will only work if
            self.cursor.execute('SELECT id, questionId FROM answer')
            result = self.cursor.fetchall()
            for row in result:
                answer_id = row[0]
                question_element_id = [row[1]]
                self.cursor.execute('SELECT id FROM question WHERE elementId=?', question_element_id)
                question_row = self.cursor.fetchone()
                question_id = question_row[0]
                values = [question_id, answer_id]
                self.cursor.execute('UPDATE answer SET questionId=? WHERE id=?', values)

            self.connection.commit()

    # Must be called before importposts
    def _import_tags(self):
        tree = ElementTree.parse(self.setting['folderprefix'] + 'Tags.xml')
        root = tree.getroot()

        # Check whether database is empty
        sql = 'SELECT * FROM tag'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        if len(result) != 0:
            logging.info('Tag table not empty, adding nothing')

        else:
            logging.info('Importing tags into tag table')
            for row in root.findall('row'):

                elementid = row.get('Id')
                tagname = row.get('TagName')
                count = row.get('Count')

                values = [elementid, tagname, count]
                self.cursor.execute('INSERT INTO tag VALUES (NULL, ?, ?, ?)', values)

            self.connection.commit()

    # Used in importposts, do not use anywhere else
    def _insert_tags_for_post(self, tags, questionid):
        regex = re.compile('(<.+?>)')
        for tag in regex.findall(tags):
            # Get tag ID
            tag = tag.replace('<', '')
            tag = tag.replace('>', '')
            tag = (tag,)
            self.cursor.execute('SELECT id FROM tag WHERE tagName=?', tag)
            row = self.cursor.fetchone()
            if row is not None:
                tagid = row[0]
                values = [questionid, tagid]
                self.cursor.execute('INSERT INTO postTag VALUES (NULL, ?, ?)', values)
            else:
                logging.info('No tag record found for ', tag)

    def _import_links(self):
        tree = ElementTree.parse(self.setting['folderprefix'] + 'PostLinks.xml')
        root = tree.getroot()

        # Check whether database is empty
        sql = 'SELECT * FROM postLink'
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        if len(result) != 0:
            logging.info('PostLink table not empty, adding nothing')

        else:
            logging.info('Importing links into postLink table')
            for row in root.findall('row'):

                elementid = row.get('Id')
                creationdate = row.get('CreationDate')
                postid = row.get('PostId')
                relatedpostid = row.get('RelatedPostId')
                linktypeid = row.get('LinkTypeId')

                values = [elementid, creationdate, postid, relatedpostid, linktypeid]
                self.cursor.execute('INSERT INTO postLink VALUES (NULL, ?, strftime(\'%s\', ?), ?, ?, ?)', values)

            self.connection.commit()

    def _import_users(self):
        tree = ElementTree.parse(self.setting['folderprefix'] + 'Users.xml')
        root = tree.getroot()

        # Check whether database is empty
        sql = '''SELECT * FROM user'''
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        if len(result) != 0:
            logging.info('User table not empty, adding nothing')

        else:
            logging.info('Importing users into user table')
            for row in root.findall('row'):

                elementid = row.get('Id')
                reputation = row.get('CreationDate')
                creationdate = row.get('CreationDate')
                displayname = row.get('DisplayName')
                lastaccessdate = row.get('LastAccessDate')
                websiteurl = row.get('WebsiteUrl')
                location = row.get('Location')
                aboutme = row.get('AboutMe')
                views = row.get('Views')
                upvotes = row.get('UpVotes')
                downvotes = row.get('DownVotes')
                profileimageurl = row.get('ProfileImageUrl')
                age = row.get('Age')
                accountid = row.get('AccountId')

                values = [elementid, reputation, creationdate, displayname, lastaccessdate, websiteurl, location,
                          aboutme, views, upvotes, downvotes, profileimageurl, age, accountid]

                self.cursor.execute('INSERT INTO user VALUES (NULL, ?, ?, strftime(\'%s\', ?), ?, strftime(\'%s\', ?), '
                                    '?, ?, ?, ?, ?, ?, ?, ?, ?)', values)

            self.connection.commit()
