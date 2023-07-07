import argparse
import numpy as np
import csv
import json
import os
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, required=True, help='original json file directory')
    parser.add_argument('--csv_dir', type=str, required=True, help='output csv file directory')
    parser.add_argument('--max_line_number', type=int, default=-1, help='max line number to process for each file, used for testing')
    args = parser.parse_args()

    comment_json_path = os.path.join(args.json_dir, 'Movies_and_TV.json')
    meta_json_path = os.path.join(args.json_dir, 'meta_Movies_and_TV.json')
    
    comment_csv_path = os.path.join(args.csv_dir, 'comments.csv')
    customer_csv_path = os.path.join(args.csv_dir, 'customers.csv')
    product_csv_path = os.path.join(args.csv_dir, 'products.csv')

    # parse meta data, output products csv
    print('Processing meta json...')
    product_ids = set()
    with open(meta_json_path) as f:
        # define output csv handles
        product_output = open(product_csv_path, 'w')
        product_writer = csv.DictWriter(product_output, fieldnames=[
            'asin', 'title', 'brand', 'description', 'imageUrl', 'rank', 'price', 'categories'
        ])
        product_writer.writeheader()
        # handle each line
        for line_idx, line in enumerate(f):
            try:
                ori_meta = json.loads(line)
                product_id = ori_meta['asin']
                descriptions = ori_meta['description']
                imageUrls = ori_meta['imageURLHighRes']
                # handle price
                price = ori_meta.get('price', '-1')
                if price == '':
                    price = '-1'
                if price.startswith('$'):
                    price = price[1:]
                try:
                    price = float(price)
                except:
                    price = -1
                # remove 'Movies & TV' from categories
                categories = ori_meta['category']
                if 'Movies & TV' in categories:
                    categories.remove('Movies & TV')
                # handle rank
                try:
                    rank_numbers = re.findall(r'\d+', ori_meta['rank'])
                    rank = int("".join(rank_numbers))
                except: # some ranks are empty list
                    rank = -1
                if not product_id in product_ids:
                    product_writer.writerow({
                        'asin': product_id,
                        'title': ori_meta['title'],
                        'brand': ori_meta['brand'],
                        'description': descriptions[0].replace('\n', '. ') if len(descriptions) else '',
                        'imageUrl': imageUrls[0] if len(imageUrls) else '',
                        'rank': ori_meta['brand'],
                        'price': price,
                        'categories': ','.join(categories),
                        'rank': rank,
                    })
                    product_ids.add(product_id)
            except Exception as e:
                print(f'Error at line #{line_idx+1}: {line}')
                print(e)
            if (line_idx + 1) % 10000 == 0:
                print(f'Processed {line_idx+1} lines') 
            if args.max_line_number > -1 and line_idx >= args.max_line_number:
                break
        # close handle
        product_output.close()

    # parse comments data, output comments & customers csv
    print('Processing comment json...')
    customer_ids = set()
    with open(comment_json_path) as f:
        # define output csv handles
        comment_output = open(comment_csv_path, 'w')
        comment_writer = csv.DictWriter(comment_output, fieldnames=[
            'overall', 'customerId', 'asin', 'summary', 'review', 'timestamp'
        ])
        comment_writer.writeheader()
        customer_output = open(customer_csv_path, 'w')
        customer_writer = csv.DictWriter(customer_output, fieldnames=[
            'customerId', 'name', 
        ])
        customer_writer.writeheader()
        # handle each line
        for line_idx, line in enumerate(f):
            try:
                ori_comment = json.loads(line)
                customer_id = ori_comment['reviewerID']
                product_id = ori_comment['asin']
                # remove row with no product record
                if product_id not in product_ids:
                    continue
                # remove null review
                review = ori_comment.get('reviewText', '')
                if (not review) or review == '':
                    continue
                comment_writer.writerow({
                    'overall': ori_comment['overall'],
                    'asin': ori_comment['asin'],
                    'customerId': ori_comment['reviewerID'],
                    'summary': ori_comment.get('summary', ''),
                    'review': review.replace('\n', '. '),
                    'timestamp': ori_comment['unixReviewTime'],
                })
                if not customer_id in customer_ids:
                    customer_writer.writerow({
                        'customerId': customer_id,
                        'name': ori_comment.get('reviewerName', 'N/A')
                    })
                    customer_ids.add(customer_id)
            except Exception as e:
                print(f'Error at line #{line_idx+1}: {line}')
                print(e)
            if (line_idx + 1) % 100000 == 0:
                print(f'Processed {line_idx+1} lines') 
            if args.max_line_number > -1 and line_idx >= args.max_line_number:
                break
        # close handle
        comment_output.close()
        customer_output.close()


if __name__ == '__main__':
    main()