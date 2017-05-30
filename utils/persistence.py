import sqlite3

from utils.constants import DATABASE


def insert_sequence(key_id, sequence, class_label):
    """
    Insert sequence data into db.
    
    :param key_id: sequence identifier.
    :param sequence: the sequence.
    :param class_label: sequence classification.
    """
    connection = sqlite3.connect(DATABASE)
    c1 = connection.cursor()
    c2 = connection.cursor()

    # Create tables if not exist
    c1.execute('''CREATE TABLE IF NOT EXISTS sequence (key_id, sequence, class_label,
                     PRIMARY KEY(sequence, class_label))''')

    try:
        c2.execute('''INSERT INTO sequence VALUES (?,?,?)''', (key_id, sequence, class_label))
    except sqlite3.IntegrityError as err:
        print(err)
    connection.commit()
    connection.close()


def is_known_sequence(key_id):
    """
    Tells whether a sequence is already in db or not.
    
    :param key_id: sequence identifier.
    :return: True if already in, False otherwise.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('''SELECT * FROM sequence WHERE key_id = ?''', (key_id,))
    for _ in cursor:
        connection.commit()
        connection.close()
        return True  # if result contains at least one tuple, return True
    connection.close()
    return False


def get_sequences_unique_labels():
    """
    Count how many distinct sequences labels are stored in db.
    
    :return: The number of distinct labels.
    """
    result = 0
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT class_label) FROM sequence''')
    for row in cursor:
        result = row[0]
    connection.close()
    return result


def get_table(table_name, limit=None):
    """
    Get all records from a given SQL table.
    
    :param table_name: a string indicating the table.
    :param limit: maximum number of rows to be returned.
    :return: a list of lists representing the records.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    if limit:
        cursor.execute('SELECT * FROM ' + table_name + 'LIMIT ?', (limit,))
    else:
        cursor.execute('SELECT * FROM ' + table_name)
    table = cursor.fetchall()
    connection.close()
    return table


def get_rows_by_label(label_name, table_name, limit=None):
    """
    Get all records related to a given label from a given SQL table.
    
    :param label_name: a string indicating the label.
    :param table_name: a string indicating the table.
    :param limit: maximum number of rows to be returned.
    :return: a list of lists representing the records.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    if limit:
        cursor.execute('SELECT * FROM ' + table_name + ' WHERE class_label =  ? LIMIT ?', (label_name, limit))
    else:
        cursor.execute('SELECT * FROM ' + table_name + ' WHERE class_label =  ?', (label_name,))
    table = cursor.fetchall()
    connection.close()
    return table


def get_training_inputs_by_label(label_name, table_name, limit=None):
    """
    Get all training inputs related to a given label from a given SQL table.
    
    :param label_name: a string indicating the label.
    :param table_name: a string indicating the table.
    :param limit: maximum number of rows to be returned.
    :return: a list of lists representing the records.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    if limit:
        cursor.execute('SELECT  sequence, class_label FROM ' + table_name +
                       ' WHERE class_label =  ? ORDER BY RANDOM() LIMIT ?',
                       (label_name, limit))
    else:
        cursor.execute('SELECT  sequence, class_label FROM ' + table_name + ' WHERE class_label =  ? ORDER BY RANDOM()',
                       (label_name,))
    table = cursor.fetchall()
    connection.close()
    return table
