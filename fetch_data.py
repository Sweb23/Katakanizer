import mysql.connector
import pandas as pd

"""
    Fetches the data from the table that has been created in the merged SQL database (merged.sql script). 
    The data is converted into a CSV file to be used in the ML model.
"""

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",       # Change if your MySQL server is remote
    user="root",   # Replace with your MySQL username
    password="azerty", # Replace with your MySQL password
    database="loanwords"    # Your database name
)


cursor = conn.cursor()

cursor.execute("SELECT * FROM merged;")

rows = cursor.fetchall()

# Get column names from cursor.description
column_names = [desc[0] for desc in cursor.description]

# Convert to a pandas DataFrame
df = pd.DataFrame(rows, columns=column_names)


cursor.close()
conn.close()

print(df)

df.to_csv("loanwords.csv", index=False)
