import json
import os
import re
import sqlite3

# 读取以.jsonl结尾的文件
folder_path = r'/home/zwmx/xm_dev/ls_project/dataset/tigerbot_law/tigerbot-laws-plugin.json'
json_data = []


with open(folder_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        json_data.append(data)

# 连接到SQLite数据库
conn = sqlite3.connect('/home/zwmx/xm_dev/ls_project/swift/scripts/ls/sqlite/law.db')
cursor = conn.cursor()

for idx, data in enumerate(json_data[:]):
    if "条 " in data['content']:
        pattern = r"([一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟萬]+条)"
        # pattern = r'(\d+条)'

        # 使用正则表达式匹配汉字数字
        match = re.search(pattern, data["content"])
        result_str = ""
        if match:
            cursor.execute(
                "INSERT INTO legal_provisions (type, title, chapter1, content) VALUES (?, ?, ?, ?)",
                (data["type"], data["title"], data["chapter1"],
                 data["content"]))

# 提交更改
conn.commit()

# 关闭连接
conn.close()


