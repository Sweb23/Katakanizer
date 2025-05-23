# Katakanizer
 ML model to convert Latin text into Katakana

## Fetch data

The repo contains a merged.sql file containing the entierety of the data. If you want to host it on a database, feel free to do so. fetch_data.py allows you to get the data from the SQL table and dump it into a CSV for training.

## How to train

The models are already available, but if you want to train them : 

python model_training.py [optionnal: light]

The light argument trains on a reduced dataset to reduce training time.

## How to run

python main.py [optionnal: light]

## Credits
[loanwords_garaigo](https://github.com/jamesohortle/loanwords_gairaigo/) for the database and the merged.sql script.