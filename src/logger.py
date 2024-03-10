import sqlite3
import datetime

class SQLiteLogger:
    def __init__(self, db_name='logger.db'):
        self.db_name = db_name
        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS logs
                         (id INTEGER PRIMARY KEY,
                          params TEXT,
                          status INTEGER,
                         mapnum INTEGER,
                         executiontime FLOAT,                            
                         timestamp TEXT)''')

    def log(self, params, status, mapnum, executiontime):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO logs (params, status, mapnum, executiontime, timestamp) VALUES (?, ?, ?, ?, ?)", (str(params), status, mapnum, executiontime ,timestamp))

    def fetch_logs(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM logs")
            logs = c.fetchall()
        return logs

if __name__ == "__main__":
    logger = SQLiteLogger()

    # Example usage
    logger.log("This is a log message.")
    logger.log("Another log message.")
    logs = logger.fetch_logs()
    for log in logs:
        print(log)
