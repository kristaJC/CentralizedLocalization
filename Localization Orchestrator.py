# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install unidecode gspread==5.12.4
# MAGIC %restart_python
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import json
import datetime
import pandas as pd
from pyspark.sql.functions import col

from localizers.general_config import CENTRALIZED_SHEET_URL, DIR
from localizers.in_game_config import INGAME_LANG_MAPS

# COMMAND ----------

# MAGIC %run "./authenticationScript"

# COMMAND ----------

gc = get_gspread_client_from_secret_old()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for any new requests

# COMMAND ----------

sh = gc.open_by_url(CENTRALIZED_SHEET_URL)
tab = sh.worksheet("Sheet1")

all_rows = tab.get_all_values()
headers = all_rows[0]
vals = all_rows[1:]

requests = spark.createDataFrame(vals, schema=headers).filter(col("Status") == "SUBMITTED")

# COMMAND ----------

requests.display()

# COMMAND ----------

if requests.count() < 1:
    dbutils.notebook.exit("Nothing to update here...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter each to different categories based on use case

# COMMAND ----------

publishing = requests.filter(col("LocType") == "Publishing").toPandas()
ingame = requests.filter(col("LocType") == "InGame").toPandas()
marketing = requests.filter(col("LocType") == "Marketing").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run for Publishing

# COMMAND ----------

pub_outputs = []

for idx, row in publishing.iterrows():
    try:
        output = dbutils.notebook.run(DIR + "Publishing-Workaround", 10000, row.to_dict())
        output = json.loads(output)
        row['Status'] = output['status']
        row['RunID'] = output['run_id']
        row['Notes'] = output.get('notes', '')
    except Exception as e:
        row['Status'] = "FAILED"
        row['RunID'] = ""
        row['Notes'] = str(e)[:500]
    row['LastStatusUpdate'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pub_outputs.append(row)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run for InGame

# COMMAND ----------

# Injects TargetLanguages from config, then calls Generic Localizer
in_game_outputs = []

for idx, row in ingame.iterrows():
    game = row.get('Game', '')
    lang_map = INGAME_LANG_MAPS.get(game)

    if lang_map is None:
        row['Status'] = "FAILED"
        row['RunID'] = ""
        row['Notes'] = f"Unknown game '{game}'. Available: {list(INGAME_LANG_MAPS.keys())}"
        row['LastStatusUpdate'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        in_game_outputs.append(row)
        continue

    row_dict = row.to_dict()
    row_dict['TargetLanguages'] = ", ".join(lang_map.keys())

    try:
        output = dbutils.notebook.run(DIR + "Generic Localizer", 10000, row_dict)
        output = json.loads(output)
        row['Status'] = output['status']
        row['RunID'] = output['run_id']
        row['Notes'] = output.get('notes', '')
    except Exception as e:
        row['Status'] = "FAILED"
        row['RunID'] = ""
        row['Notes'] = str(e)[:500]
    row['LastStatusUpdate'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    in_game_outputs.append(row)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run for Marketing

# COMMAND ----------

marketing_outputs = []

for idx, row in marketing.iterrows():
    try:
        output = dbutils.notebook.run(DIR + "Generic Localizer", 10000, row.to_dict())
        output = json.loads(output)
        row['Status'] = output['status']
        row['RunID'] = output['run_id']
        row['Notes'] = output.get('notes', '')
    except Exception as e:
        row['Status'] = "FAILED"
        row['RunID'] = ""
        row['Notes'] = str(e)[:500]
    row['LastStatusUpdate'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marketing_outputs.append(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine and write back to tracking sheet

# COMMAND ----------

updates = []
if len(pub_outputs) > 0:
    updates.append(pd.DataFrame(pub_outputs))
if len(in_game_outputs) > 0:
    updates.append(pd.DataFrame(in_game_outputs))
if len(marketing_outputs) > 0:
    updates.append(pd.DataFrame(marketing_outputs))

if len(updates) > 1:
    updates = pd.concat(updates)
elif len(updates) == 1:
    updates = updates[0]
else:
    print("nothing to update!!")
    dbutils.notebook.exit("Nothing to see here")

# COMMAND ----------

updates

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the cells in the centralized tracking sheet

# COMMAND ----------

for i, row in updates.iterrows():
    cell = tab.find(row['RowFingerprint'])
    status_range = f"I{cell.row}"   # status
    last_updated  = f"J{cell.row}"  # last updated
    run_id        = f"L{cell.row}"  # run_id
    notes         = f"P{cell.row}"  # notes

    tab.update_acell(status_range, row['Status'])
    tab.update_acell(last_updated, row['LastStatusUpdate'])
    tab.update_acell(run_id, row['RunID'])
    tab.update_acell(notes, row.get('Notes', ''))
