# HSBCQuizDataML
Data &amp; ML code base from HSBC quiz

## Data Preprocessing

- Understand original data
- Convert JSON data to 3 csv files (products, customers & comments)
- Further understand & experiment on the data using Pandas, SpaCy, sklearn, etc., mainly on a subset
- Augment comment data with semantic analysis & word count
- Key word detection on comment data, to form word cloud
- Augment product data with review count, avg rating, avg sentiment & avg conbined rating

The pipeline for the data preprocessing phase will be e.g.:

```cmd
// convert json to csv
python convert_json_to_csv.py --json_dir ../data/original --csv_dir ../data/csv
// spacy functions, incl. semantic anaylysis, word count & tokenization
python process_with_spacy.py --comment_path ../data/csv/comments.csv --customer_cluster_base ../data/customer_cluster --aug_comment_path ../data/csv/aug_comments.csv --keyword_path ../data/csv/keywords.csv --batch_size 1000
// augment products with comment info
python augment_products.py --comment_csv_path ../data/csv/aug_comments.csv --product_csv_path ../data/csv/products.csv --aug_product_csv_path ../data/csv/aug_products.csv
```

## Clustering on Customer Comments

- Gather all comments on each customer, and calculate their vector representation (done in data preprocess with spacy)
- Use MiniBatchKMeans to cluster, with elbow method to select an appropriate K
- Understand the data in Jupyter Notebook, with visualization & selected comments based on the cluster, on a subset of data
- Augment customer data with cluster id (based on all their comments), distance to cluster center and 2d decomposition (for visualization)
- Save the clustering model & pca model with Joblib

The cmd for this task e.g.:

```cmd
python cluster.py --customer_csv_path ../data/csv/customers.csv --vector_path ../data/customer_cluster/customer_vectors.npy --customer_id_path ../data/customer_cluster/customer_ids.txt  --aug_customer_csv_path ../data/csv/aug_customers.csv --output_model_dir ../models/cluster/ --k 6 --batch_size 1024
```

# LoRA

- Build train pipeline including dataset loading & preprocessing, train model defining (with LoRA using PEFT) and evaluation
- A Jupyter Notebook for inference testing

The cmd for this task e.g.:

```
python train.py --comment_path ../data/csv/aug_comments.csv --output_dir ../models/amz_movie_tv_distilgpt2_50k_longest --sample_method longest --max_line_number 50000
```