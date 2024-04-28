import sqlite3

# 连接到SQLite数据库（如果不存在则会创建）
conn = sqlite3.connect('/home/zwmx/xm_dev/ls_project/swift/scripts/ls/sqlite/law.db')

# 创建一个游标对象
cursor = conn.cursor()

# 创建一个表
# 创建表

cursor.execute('''CREATE TABLE legal_provisions
                (id INTEGER PRIMARY KEY,
                type TEXT,
                title TEXT,
                chapter1 TEXT,
                content TEXT)''')
# # 插入一条数据
# cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")

# 提交更改
conn.commit()

# 关闭连接
conn.close()

print("SQLite数据库和表创建成功")