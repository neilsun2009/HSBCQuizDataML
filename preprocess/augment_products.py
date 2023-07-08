import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import joblib

# augment product with comment info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_csv_path', type=str, required=True, help='input csv file path for comments')
    parser.add_argument('--product_csv_path', type=str, required=True, help='input csv file path for products')
    parser.add_argument('--aug_product_csv_path', type=str, required=True, help='output csv file for augmented info on products')
    parser.add_argument('--max_comments', type=int, default=-1, help='max amount of comments to calculate, use for test mode')
    args = parser.parse_args()

    # load csv files
    print('Loading product csv...')
    product_df = pd.read_csv(args.product_csv_path)
    print(f'{len(product_df)} products')
    print('Loading comment csv...')
    comment_df = pd.read_csv(args.comment_csv_path, nrows=None if args.max_comments == -1 else args.max_comments)
    print(f'{len(comment_df)} comments')
    
    print('Group-by on comments')
    comment_df = comment_df[['asin', 'overall', 'sentiment', 'sentimentWithRating']]
    group_by_df = comment_df.groupby('asin')
    aug_df = group_by_df.mean()
    count_df = group_by_df.size().reset_index(name='reviewCount')
            
    aug_df.rename(columns={
        'sentiment': 'avgSentiment',
        'overall': 'avgOverall',
        'sentimentWithRating': 'avgSentimentWithRating',
    }, inplace=True)
    
    print('Merging comment file...')
    product_df = pd.merge(product_df, aug_df, how='left', on="asin")
    product_df = pd.merge(product_df, count_df, how='left', on="asin")
    product_df['avgOverall'].fillna(0.0, inplace=True)
    product_df['avgSentiment'].fillna(0.0, inplace=True)
    product_df['avgSentimentWithRating'].fillna(0.0, inplace=True)
    product_df['reviewCount'].fillna(0, inplace=True)
    product_df['reviewCount'] = product_df['reviewCount'].astype(int)
    
    print('Wrting output...')
    product_df.to_csv(args.aug_product_csv_path, index=False)


if __name__ == '__main__':
    main()