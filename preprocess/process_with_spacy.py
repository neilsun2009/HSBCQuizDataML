import argparse
import numpy as np
import pandas as pd
import csv
import json
import os
import re
from tqdm import tqdm
import spacy
import string
from spacytextblob.spacytextblob import SpacyTextBlob

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")
nlp.add_pipe('spacytextblob')

def cal_doc_vector(text):
    doc = nlp(text)
    return doc.vector

def cal_sentiment(text):
    # -1 to 1
    doc = nlp(text)
    return doc._.blob.polarity

def count_words(text):
    doc = nlp(text.lower())
    word_count_dict = dict()
    for token in doc:
        word = token.text
        if word in string.punctuation:
            continue
        if word not in word_count_dict:
            word_count_dict[word] = {
                'count': 0,
                'is_keyword': False
            }
        word_count_dict[word]['count'] += 1
        if token.pos_ in ['PROPN', 'ADJ', 'NOUN']:
            word_count_dict[word]['is_keyword'] = True
    return word_count_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_path', type=str, required=True, help='input csv file path for all comments')
    parser.add_argument('--aug_comment_path', type=str, required=True, help='output csv file path for all comments with augmentation')
    parser.add_argument('--keyword_path', type=str, required=True, help='output csv file path for keywords')
    parser.add_argument('--customer_cluster_base', type=str, required=True, help='output dir for customer cluster files')
    parser.add_argument('--max_line_number', type=int, default=-1, help='max line number to process, used for testing')
    args = parser.parse_args()

    customer_vectors_path = os.path.join(args.customer_cluster_base, 'customer_vectors.npy')
    customer_ids_path = os.path.join(args.customer_cluster_base, 'customer_ids.txt')

    comment_df = pd.read_csv(args.comment_path, nrows=None if args.max_line_number == -1 else args.max_line_number)
    # sort by customer id, used to calculate comment vector for each customer
    comment_df.sort_values(by=['customerId'], inplace=True)

    print('CSV read & sorted')

    lastCustomerId = ''
    lastCustomerReview = ''
    # handles for customer cluster related files
    customer_vectors_file = open(customer_vectors_path, 'wb')
    customer_ids_file = open(customer_ids_path, 'w')

    sentiments = []
    word_counts = []
    word_counter = {}
    with tqdm(total=len(comment_df)) as pbar:
        for idx, comment in comment_df.iterrows():
            review = comment['review']
            customerId = comment['customerId']
            # calculate vector of last customer's review
            if customerId != lastCustomerId:
                doc_vector = cal_doc_vector(lastCustomerReview)
                customer_ids_file.write(f'{customerId}\n')
                np.save(customer_vectors_file, doc_vector)
                lastCustomerId = customerId
                lastCustomerReview = ''
            # append review for the same customer
            lastCustomerReview += '. ' + review
            # calculate sentiment
            sentiment = cal_sentiment(review)
            sentiments.append(sentiment)
            # calculate word count 
            word_count_dict = count_words(review)
            word_counts.append(len(word_count_dict))
            for word, attrs in word_count_dict.items():
                # only count keywords
                if not attrs['is_keyword']:
                    continue
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += attrs['count']
            pbar.update(1)


    # calculate last customer's review vector
    doc_vector = cal_doc_vector(lastCustomerReview)
    customer_ids_file.write(f'{customerId}\n')
    np.save(customer_vectors_file, doc_vector)
    # close
    customer_vectors_file.close()
    customer_ids_file.close()

    # write new columns to comments
    comment_df['sentiment'] = sentiments
    # avg score from sentiment and overall rating
    # 0-10
    comment_df['sentimentWithRating'] = (np.array(sentiments) + 1) * 2.5 + comment_df['overall']
    # word counts in the review
    comment_df['wordCount'] = word_counts
    comment_df.to_csv(args.aug_comment_path, index=False)

    # write keywords counts to csv
    with open(args.keyword_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'count',
        ])
        writer.writeheader()
        for word, count in word_counter.items():
            writer.writerow({
                'word': word,
                'count': count
            })

if __name__ == '__main__':
    main()