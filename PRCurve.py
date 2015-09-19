import sys

theme = ''
model = ''

if len(sys.argv) > 2:
    theme = sys.argv[1]
    model = sys.argv[2]
else:
    print('Given arguments do not fit')
    sys.exit(0)


def write_curve_for_topic_model(topic_model):
    directory = "/Users/Cordt/Documents/results/" + theme + "/"
    precision_file = theme + '_precision_' + topic_model + '.txt'
    precision_path = ''.join([directory, precision_file])
    recall_file = theme + '_recall_' + topic_model + '.txt'
    recall_path = ''.join([directory, recall_file])

    output_file = 'PR_' + theme + '_' + topic_model + '.dat'
    output_path = ''.join([directory, output_file])

    with open(precision_path, "r") as f:
        content_precision = f.readlines()

    with open(recall_path, "r") as f:
        content_recall = f.readlines()

    file_handler = open(output_path, "w")
    for (index, line) in enumerate(content_recall):
        key = content_recall[index].split()[1]
        value = content_precision[index].split()[1]
        file_handler.write(str(key) + ', ' + str(value) + '\n')


def write_curve_for_tfidf():
    directory = "/Users/Cordt/Documents/results/" + theme + "/"
    precision_file = theme + '_precision_tf-idf.txt'
    precision_path = ''.join([directory, precision_file])
    recall_file = theme + '_recall_tf-idf.txt'
    recall_path = ''.join([directory, recall_file])

    output_file = 'PR_' + theme + '_tfidf.dat'
    output_path = ''.join([directory, output_file])

    with open(precision_path, "r") as f:
        content_precision = f.readlines()

    with open(recall_path, "r") as f:
        content_recall = f.readlines()

    file_handler = open(output_path, "w")
    for (index, line) in enumerate(content_recall):
        key = content_recall[index].split()[1]
        value = content_precision[index].split()[1]
        file_handler.write(str(key) + ', ' + str(value) + '\n')

if model == 'topics':
    write_curve_for_topic_model('answers')
    write_curve_for_topic_model('questions')
    write_curve_for_topic_model('qanda')

elif model == 'tfidf':
    write_curve_for_tfidf()
