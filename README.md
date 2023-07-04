# HSBCQuizDataML
Data &amp; ML code base from HSBC quiz

## Data Preprocessing

- Understand original data
- Convert JSON data to 3 csv files (products, customers & comments)
- Further understand & experiment on the data using Pandas, SpaCy, sklearn, etc., mainly on a subset
- Augment comment data with semantic analysis & word count
- Augment customer data with clustering on comments
- Key word detection on comment data, to form word cloud

The pipeline for the data preprocessing phase will be e.g.:

```
// convert json to csv
python convert_json_to_csv.py --json_dir ../data/original --csv_dir ../data/csv
// spacy functions, incl. semantic anaylysis, word count & tokenization
python process_with_spacy.py --comment_path ../data/csv/comments.csv --customer_cluster_base ../data/customer_cluster --aug_comment_path ../data/csv/aug_comments.csv --keyword_path ../data/csv/keywords.csv
```