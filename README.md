# Project 7 Open-Ended Project

Pyspark VS Pandas

### Enviroment Requirement

- conda create -n CS5239 python=3.8 pandas scikit-learn jupyter pyspark=3.4 spark-nlp openjdk=8
- Scala 2.12
- Maven 3.5

## Task 1

### Task 
Write a Spark program to find the top 15 products based on the number of reviews each day and report their average ratings, review time and product brand name.

### Dataset
Use the Patio Lawn and Garden review file (Patio_Lawn_and_Garden.json, 2.44GB) and metadata (meta_Patio_Lawn_and_Garden.json) from the [Amazon product dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/, 4.43GB).
Download both files from the “Per-category files” section.

Input: Review file and metadata.
Output: One line per product in the following format:

`<product ID > <num of reviews> <review time> <avg ratings> <product brand name>`



## Task 2

### Task
Movie review sentiment analysis

A big data analysis pipeline based on utilizing language modeling and text classification techniques, to process and analyze large-scale movie review datasets,

### Dataset
movie.csv (52.7MB)

The dataset contains a total of 39,273 data items, with each review accompanied by a sentiment label indicating the reviewer’s sentiment polarity towards the movie.