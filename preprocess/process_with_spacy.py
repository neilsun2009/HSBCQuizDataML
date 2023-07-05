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
nlp = spacy.load("en_core_web_md", disable=['parser', 'ner', 'lemmatizer', 'textcat'])
nlp.add_pipe('spacytextblob')

word_counter = {}

def cal_doc_vector(doc):
    # doc = nlp(text)
    return doc.vector

def cal_sentiment(doc):
    # -1 to 1
    # doc = nlp(text)
    return doc._.blob.polarity

def count_words(doc):
    # doc = nlp(text.lower())
    words = set()
    for token in doc:
        word = token.text
        if word in string.punctuation:
            continue
        words.add(word)
        if token.pos_ not in ['PROPN', 'ADJ', 'NOUN']:
            continue
        if word not in word_counter:
            word_counter[word] = 0
        word_counter[word] += 1
    return len(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_path', type=str, required=True, help='input csv file path for all comments')
    parser.add_argument('--aug_comment_path', type=str, required=True, help='output csv file path for all comments with augmentation')
    parser.add_argument('--keyword_path', type=str, required=True, help='output csv file path for keywords')
    parser.add_argument('--customer_cluster_base', type=str, required=True, help='output dir for customer cluster files')
    parser.add_argument('--max_line_number', type=int, default=-1, help='max line number to process, used for testing')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for spacy to process')
    parser.add_argument('--max_text_length', type=int, default=800000, help='max length of text to be handled by spacy')
    args = parser.parse_args()

    customer_vectors_path = os.path.join(args.customer_cluster_base, 'customer_vectors.npy')
    customer_ids_path = os.path.join(args.customer_cluster_base, 'customer_ids.txt')

    comment_df = pd.read_csv(args.comment_path, nrows=None if args.max_line_number == -1 else args.max_line_number)
    # sort by customer id, used to calculate comment vector for each customer
    comment_df.sort_values(by=['customerId'], inplace=True)

    print('CSV read & sorted')

    lastCustomerId = comment_df.iloc[0]['customerId']
    lastCustomerReview = ''
    # handles for customer cluster related files
    customer_vectors_file = open(customer_vectors_path, 'wb')
    customer_ids_file = open(customer_ids_path, 'w')

    sentiments = []
    word_counts = []

    comment_batch = []
    customer_batch = []
    with tqdm(total=len(comment_df)//args.batch_size) as pbar:
        for idx, comment in comment_df.iterrows():
            review = str(comment['review'])
            customerId = comment['customerId']
            # calculate vector of last customer's review
            if customerId != lastCustomerId:
                customer_batch.append(lastCustomerReview[:min(args.max_text_length, len(lastCustomerReview))])
                if len(customer_batch) == args.batch_size:
                    for doc in nlp.pipe(customer_batch, disable=['tagger', 'spacytextblob']):
                        doc_vector = cal_doc_vector(doc)
                        np.save(customer_vectors_file, np.array(doc_vector.get() if spacy.prefer_gpu() else doc_vector))
                    customer_batch = []
                customer_ids_file.write(f'{lastCustomerId}\n')
                lastCustomerReview = ''
            # append review for the same customer
            lastCustomerReview += review + '. '
            lastCustomerId = customerId
            # calculate in batch
            comment_batch.append(review.lower()[:min(args.max_text_length, len(review))])
            if len(comment_batch) == args.batch_size:
                for doc in nlp.pipe(comment_batch):
                    # calculate sentiment
                    sentiment = cal_sentiment(doc)
                    sentiments.append(sentiment)
                    # calculate word count 
                    word_count = count_words(doc)
                    word_counts.append(word_count)
                comment_batch = []
                pbar.update(1)


    # calculate last customer's review vector
    customer_batch.append(lastCustomerReview[:min(args.max_text_length, len(lastCustomerReview))])
    customer_ids_file.write(f'{lastCustomerId}\n')
    for doc in nlp.pipe(customer_batch, disable=['tagger', 'spacytextblob']):
        doc_vector = cal_doc_vector(doc)
        np.save(customer_vectors_file, np.array(doc_vector.get() if spacy.prefer_gpu() else doc_vector))
    # close
    customer_vectors_file.close()
    customer_ids_file.close()

    # calculate last batch of comments
    if len(comment_batch):
        for doc in nlp.pipe(comment_batch):
            # calculate sentiment
            sentiment = cal_sentiment(doc)
            sentiments.append(sentiment)
            # calculate word count 
            word_count = count_words(doc)
            word_counts.append(word_count)
        comment_batch = []

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