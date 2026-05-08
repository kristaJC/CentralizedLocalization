# Databricks notebook source

# COMMAND ----------

import json
import gspread


def get_gspread_client_from_secret_old(scope='krista-ds-loc-scope', key='crafty_json_key'):
    try:
        crafty_json = dbutils.secrets.get(scope='krista-ds-loc-scope', key='crafty_json_key')
    except:
        print("Cannot get secret from scope for gspread!!")

    gc = gspread.service_account_from_dict(json.loads(crafty_json))
    print("Getting gspread client...")
    return gc


def get_gspread_client_from_secret(scope='krista-ds-gspread-access', key='authentication'):
    auth_json = dbutils.secrets.get(scope=scope, key=key)
    gc = gspread.service_account_from_dict(json.loads(auth_json))
    print("Getting gspread client...")
    return gc


def get_gpt_secret_keys():
    try:
        current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split("@")[0]
        scope_name = f'openai-{current_user}'
        api_user_key = dbutils.secrets.get(scope=scope_name, key="token")
    except Exception as e:
        raise Exception(str(e))

    try:
        dbutils.secrets.list(scope="openai")
        org_key = dbutils.secrets.get(scope="openai", key="token_org")
        return (current_user, api_user_key, org_key)
    except Exception as e:
        raise Exception(str(e))


def get_model_client(model_type: str = 'gpt'):
    if model_type not in ['gpt', 'gemini']:
        print(f"The model type: {model_type} is not supported. Please choose either 'gpt' or 'gemini'")
        return None

    if model_type == 'gpt':
        CURRENT_USER, PROJECT_API_KEY, ORGANIZATION_API_KEY = get_gpt_secret_keys()
        from openai import OpenAI
        client = OpenAI(api_key=PROJECT_API_KEY, organization=ORGANIZATION_API_KEY)
        print('Getting gpt client...')
        return client

    return None


def setup_widgets():
    dbutils.widgets.text('RowFingerprint', "")
    dbutils.widgets.text("Timestamp", "")
    dbutils.widgets.text("SubmitterEmail", "")
    dbutils.widgets.text("DueDate", "")
    dbutils.widgets.text('LocType', "")
    dbutils.widgets.text("Game", "")
    dbutils.widgets.text("TargetLanguages", "")
    dbutils.widgets.text("URL", "")
    dbutils.widgets.text("QAFlag", "")
    dbutils.widgets.text("Status", "")
    dbutils.widgets.text("LastStatusUpdate", "")


def get_request():
    return {
        'Timestamp': dbutils.widgets.get('Timestamp'),
        'SubmitterEmail': dbutils.widgets.get('SubmitterEmail'),
        'DueDate': dbutils.widgets.get("DueDate"),
        'Game': dbutils.widgets.get("Game"),
        'TargetLanguages': dbutils.widgets.get("TargetLanguages"),
        'URL': dbutils.widgets.get("URL"),
        'LocType': dbutils.widgets.get("LocType"),
        'RowFingerprint': dbutils.widgets.get("RowFingerprint"),
    }


setup_widgets()
