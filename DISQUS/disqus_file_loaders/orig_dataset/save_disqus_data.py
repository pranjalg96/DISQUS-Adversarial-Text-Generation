import csv
import random

# Will store all comments in the DISQUS dataset. Split into valid and test after generation.
comments = []

avg_length = 0.0
max_length = 0

num_selected_comments = 0
num_toxic_comments = 0  # Number of toxic comments within the selected set of comments after truncating

first_row = True
trunc_len = 60

comment_idx = 2
toxic_label_idx = 3

# Read csv file and filter out comments
with open('DISQUS/disqus_dataset/disqus_data.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel')

    for row in csvreader:

        if first_row:
            first_row = False
            continue

        comment = row[comment_idx]
        toxic_label = row[toxic_label_idx]

        comment_word_list = comment.split(' ')
        len_i = len(comment_word_list)

        if len_i > trunc_len:
            continue

        if toxic_label == '1':
            num_toxic_comments += 1

        if len_i > max_length:
            max_length = len_i

        avg_length += len_i

        comments.append([comment])
        num_selected_comments += 1

avg_length /= num_selected_comments

print('Average length of comments: ', avg_length)
print('Max length of comments: ', max_length)

print('Number of selected comments: ', num_selected_comments)
print('Number of toxic comments: ', num_toxic_comments)

train_examples = comments

# Create training file
with open('DISQUS/disqus_dataset/disqus_data_filtered.csv', 'w', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel', delimiter='"')

    for row in train_examples:

        wr.writerow(row)










