import pandas as pd
import numpy as np
import matplotlib as mp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import os
import argparse
from tqdm import tqdm
import joblib

# steps
# perform clustering
# record cluster id, distance to center, 2d representation of each customer
# save the clustering model

RANDOM_STATE = 16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--customer_csv_path', type=str, required=True, help='input csv file path for customers')
    parser.add_argument('--vector_path', type=str, required=True, help='input numpy file for vectors')
    parser.add_argument('--customer_id_path', type=str, required=True, help='input text file for mapping between line no. and customer id')
    parser.add_argument('--k', type=int, default=6, help='number of clusters')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for cluster methods')
    parser.add_argument('--aug_customer_csv_path', type=str, required=True, help='output csv file for augmented info on customers')
    parser.add_argument('--output_model_path', type=str, required=True, help='output path for kmeans model')
    parser.add_argument('--max_data_lines', type=int, default=-1, help='max amount of data lines to be handled')
    args = parser.parse_args()
    
    print('Loading vector file...')
    vector_data = []
    with open(args.vector_path, 'rb') as f:
        if args.max_data_lines == -1:
            while True:
                try:
                    vector_data.append(np.load(f))
                except:
                    break
        else:
            for i in range(args.max_data_lines):
                vector_data.append(np.load(f))

    print('Calculating PCA...')
    pca = PCA(2)
    pca_data = pca.fit_transform(vector_data)

    print('Calculating MiniBatchKMeans...')
    kmeans = MiniBatchKMeans(
        n_clusters=args.k, 
        random_state=RANDOM_STATE,
        max_iter=100,
        batch_size=args.batch_size,
    )
    labels = kmeans.fit_predict(vector_data)
    print(f'Inertia: {kmeans.inertia_}')

    print('Calculating distances...')
    customer_ids = []
    distances = []
    with open(args.customer_id_path) as f:
        with tqdm(total=len(labels)) as pbar:
            for idx, line in enumerate(f):
                if args.max_data_lines != -1 and idx >= args.max_data_lines:
                    break
                customer_id = line.strip()
                customer_ids.append(customer_id)
                distances.append(np.linalg.norm(vector_data[idx] - kmeans.cluster_centers_[labels[idx]]))
                pbar.update(1)
    del vector_data

    print('Building new dataframe...')
    aug_df = pd.DataFrame()
    aug_df['customerId'] = customer_ids
    aug_df['cluster'] = labels
    aug_df['cluster'] = aug_df['cluster'].astype(int)
    aug_df['distanceToCenter'] = distances
    aug_df['vector2dX'] = pca_data[:, 0]
    aug_df['vector2dY'] = pca_data[:, 1]

    print('Merging comment file...')
    customer_df = pd.read_csv(args.customer_csv_path, nrows=None if args.max_data_lines == -1 else args.max_data_lines)
    
    customer_df = pd.merge(customer_df, aug_df, how='left', 
                            left_on="customerId", right_on="customerId")

    print('Wrting output...')
    customer_df.to_csv(args.aug_customer_csv_path, index=False)
    joblib.dump(kmeans, args.output_model_path)


if __name__ == '__main__':
    main()