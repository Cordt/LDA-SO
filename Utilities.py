import sqlite3

####################################################################################################
# Utilitiy methods
####################################################################################################


def load_similarities_for_question(result_folder_path, similarity_tablename, question_id, ordered_ascending=True,
                                   order_by='similarity'):
    filename = "/model/similarities.db"
    dbpath = ''.join([result_folder_path, filename])

    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    values = [question_id]
    if ordered_ascending:
        # The smaller the closer --> ascending
        cursor.execute('SELECT questionId, answerId, similarity FROM ' +
                       similarity_tablename + ' WHERE questionId=? ORDER BY ' + order_by + ' ASC', values)
    else:
        # The larger the closer --> descending
        cursor.execute('SELECT questionId, answerId, similarity FROM ' +
                       similarity_tablename + ' WHERE questionId=? ORDER BY ' + order_by + ' DESC', values)

    return cursor.fetchall()


def load_related_answer_similarities_for_question(theme_dbpath, result_folder_path, similarity_tablename, question_id):
    filename = "/model/similarities.db"
    dbpath = ''.join([result_folder_path, filename])

    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    related_answers_ids = get_related_answer_ids(theme_dbpath, question_id, with_score=False)
    related_answers_id_strings = []
    for element in related_answers_ids:
        related_answers_id_strings.append(str(element))
    tmp_string = ','.join(related_answers_id_strings)

    sql = 'SELECT answerId, similarity FROM ' + similarity_tablename + ' WHERE questionId=' + \
          str(question_id) + ' AND answerId IN (' + tmp_string + ') ORDER BY similarity ASC'

    # The smaller the closer --> ascending
    cursor.execute(sql)

    result = cursor.fetchall()
    related_answer_similarities = []
    for row in result:
        # Store a tuple like (answer ID, similarity)
        related_answer_similarities.append((row[0], row[1]))

    return related_answer_similarities


def load_related_answer_lengths(theme_dbpath, result_folder_path, question_id):
    filename = "/model/lenghts.db"
    dbpath = ''.join([result_folder_path, filename])

    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    related_answers_ids = get_related_answer_ids(theme_dbpath, question_id, with_score=False)
    related_answers_id_strings = []
    for element in related_answers_ids:
        related_answers_id_strings.append(str(element))
    tmp_string = ','.join(related_answers_id_strings)

    # The smaller the closer --> ascending
    cursor.execute('SELECT answerId, length FROM `lengths` WHERE '
                   'answerId IN (' + tmp_string + ') ORDER BY length DESC')

    result = cursor.fetchall()
    related_answer_lenghts = []
    for row in result:
        # Store a tuple like (answer ID, length)
        related_answer_lenghts.append((row[0], row[1]))

    return related_answer_lenghts


def get_max_question_id(result_folder_path, similarities_table_name):
    filename = "/model/similarities.db"
    dbpath = ''.join([result_folder_path, filename])

    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    cursor.execute('SELECT MAX(questionId) FROM ' + similarities_table_name)
    return cursor.fetchone()[0]


def get_related_answer_ids(theme_dbpath, question_id, with_score=False):

    # Database connection - instance variables
    connection = sqlite3.connect(theme_dbpath)
    cursor = connection.cursor()

    answer_ids = []

    # Get the related answers and their scores
    values = [question_id]
    cursor.execute('SELECT id, score FROM answer WHERE questionId=? ORDER BY score DESC', values)
    related_answer_element_ids = cursor.fetchall()

    # Translate answer element ID's to ID's
    for answer_element_id in related_answer_element_ids:
        if with_score:
            # Store a tuple like (answer ID, score)
            answer_ids.append((answer_element_id[0], answer_element_id[1]))
        else:
            answer_ids.append(answer_element_id[0])

    return answer_ids


def get_number_of_related_answers(theme_dbpath):
    # Database connection - instance variables
    connection = sqlite3.connect(theme_dbpath)
    cursor = connection.cursor()

    number_of_related_answers = []

    cursor.execute('SELECT id, answerCount FROM question ORDER BY score DESC')
    result = cursor.fetchall()

    for element in result:
        # Store a tuple like (question ID, number of answers)
        number_of_related_answers.append(element[1])

    return number_of_related_answers


def create_clean_similarities_table(result_folder_path, similarities_table_name, clean=False):
    filename = "/model/similarities.db"
    dbpath = ''.join([result_folder_path, filename])
    print(dbpath)
    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    # Drop table if required
    if clean:
        sql = 'DROP TABLE IF EXISTS ' + similarities_table_name
        cursor.execute(sql)

    sql = 'CREATE TABLE IF NOT EXISTS ' + similarities_table_name + \
          ' (questionId int, answerId int, similarity real, ' \
          'PRIMARY KEY (questionId, answerId))'
    cursor.execute(sql)


def write_similarities_to_db(result_folder_path, similarities_table_name, similarities):
    filename = "/model/similarities.db"
    dbpath = ''.join([result_folder_path, filename])

    # Database connection - instance variables
    connection = sqlite3.connect(dbpath)
    cursor = connection.cursor()

    values = []
    for (question_id, answer_id, similarity) in similarities:
        values.append((question_id, answer_id, similarity))

    cursor.executemany('INSERT INTO ' + similarities_table_name + ' VALUES (?, ?, ?)', values)

    connection.commit()


def sorted_similarity_list(similarities):
    return sorted(similarities, key=lambda similarity: similarity[2])
