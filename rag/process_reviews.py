# written specifically to extract reviews from the csv file here - https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download
from argparse import ArgumentParser
import os

import csv


def main():
    parser = ArgumentParser(description='Search index for text files')
    parser.add_argument('--reviews_file', '-f', required=True, help='File with reviews')
    parser.add_argument('--target_path', '-p', required=True, help='Path to put files')

    args = parser.parse_args()

    with open(args.reviews_file, "r") as reviews:
        reader = csv.DictReader(reviews)

        for review in reader:
            review_id = review["Id"]
            text = review["Text"]
            summary = review["Summary"]
            score = review["Score"]
            product_id = review["ProductId"]

            file_path = f"{args.target_path}/{int(review_id) % 1000}"
            os.makedirs(file_path, exist_ok=True)

            with open(f"{file_path}/{review_id}.txt", "w") as review_file:
                review_file_contents = f"Product Id: {product_id}\nRating: {score}\nSummary: {summary}\nText: {text}\n"
                review_file.write(review_file_contents)
            print(f"Processed: {review_id}")


if __name__ == "__main__":
    main()
