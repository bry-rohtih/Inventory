import sqlite3 as sql
import pandas as pd
from datetime import date


conn = sql.connect("inventory.db",check_same_thread=False)

# create table
def create():
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS DATA(Time DATE, Name TEXT, Count INT)''')
    conn.commit()

# adding data to the table
def Insert(date, text, count):
    conn.execute('''INSERT INTO DATA VALUES(?,?,?)''',
                 (date, text, count))
    conn.commit()

# read contents of the data
def read():
    result = []
    state = conn.execute('''SELECT * FROM DATA''')
    data = state.fetchall()
    for row in data:
        result.append(row)
    df = pd.DataFrame(result,columns=['time','name','count'])
    return df

def readtable2():
    l = []
    state = conn.execute('''SELECT Name,sum(Count) FROM DATA GROUP BY name''')
    data = state.fetchall()
    for row in data:
        l.append(row)
    df = pd.DataFrame(l,columns=['','count'])
    
    return df

# def rename():
#     conn.execute('''ALTER TABLE ITEMS
#   RENAME TO DATA;''')
#     conn.commit()

# deleting table
# def _delete():
#     conn.execute('''DELETE FROM ITEMS''')
#     conn.commit()

if __name__ == '__main__':
    create()    
    
    
    # Insert(date.today(),'Apples',42)
    # Insert(date.today(),'Mangoes',31)
    # Insert(date.today(),'Bananas',53)
    # Insert(date.today(),'Pine Apples',23)
    # Insert(date.today(),'Coke',61)
    # Insert(date.today(),'Sprite',83)
