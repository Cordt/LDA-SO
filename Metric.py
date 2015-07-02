__author__ = 'Cordt'

from scipy import stats
import numpy as np


def js_distance(p, q):
    return np.sqrt(js_divergence(p, q))


def js_divergence(p, q):
    _p = p / np.linalg.norm(p, ord=1)
    _q = q / np.linalg.norm(q, ord=1)
    _m = 0.5 * (_p + _q)
    return 0.5 * (stats.entropy(_p, _m) + stats.entropy(_q, _m))


def exact_match_distance(score_ordered_answers, similarity_ordered_answers):

        # score_ordered_answers - (answer ID, score)
        # similarity_ordered_answers - (answer ID, similarity)

        no_of_answers = len(score_ordered_answers)
        if no_of_answers != 0:
            average_distance = 0.0
            for (outer_answer_index, (outer_anwser_id, _)) in enumerate(score_ordered_answers):
                if outer_anwser_id != similarity_ordered_answers[outer_answer_index][0]:
                    average_distance += float(1.0)

            # Normalization
            if no_of_answers == 1:
                return -1
            else:
                average_distance *= (1.0 / float(no_of_answers))
            return average_distance

        else:
            return -1

# Spearman's footrule
def deviation_distance(score_ordered_answers, similarity_ordered_answers):

        # score_ordered_answers - (answer ID, score)
        # similarity_ordered_answers - (answer ID, similarity)

        no_of_answers = len(score_ordered_answers)
        if no_of_answers != 0:
            average_distance = 0.0
            for (outer_answer_index, (outer_anwser_id, _)) in enumerate(score_ordered_answers):
                for (inner_answer_index, (inner_answer_id, __)) in enumerate(similarity_ordered_answers):
                    if outer_anwser_id == inner_answer_id:
                        average_distance += float(abs(outer_answer_index - inner_answer_index))
                        break
                    else:
                        continue

            # Normalization
            if no_of_answers == 1:
                return -1
            elif float(no_of_answers) % 2 == 0:
                average_distance *= (2.0 / float(pow(no_of_answers, 2)))
            else:
                average_distance *= (2.0 / float(pow(no_of_answers, 2) - 1))
            return average_distance

        else:
            return -1

def squared_deviation_distance(score_ordered_answers, similarity_ordered_answers):

        # score_ordered_answers - (answer ID, score)
        # similarity_ordered_answers - (answer ID, similarity)

        no_of_answers = len(score_ordered_answers)
        if no_of_answers != 0:
            average_distance = 0.0
            for (outer_answer_index, (outer_anwser_id, _)) in enumerate(score_ordered_answers):
                for (inner_answer_index, (inner_answer_id, __)) in enumerate(similarity_ordered_answers):
                    if outer_anwser_id == inner_answer_id:
                        average_distance += float(pow(outer_answer_index - inner_answer_index, 2))
                        break
                    else:
                        continue

            # Normalization
            if no_of_answers == 1:
                return -1
            else:
                average_distance *= (3.0 / float(pow(no_of_answers, 3) - no_of_answers))
            return average_distance

        else:
            return -1
