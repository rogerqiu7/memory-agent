"""Small boto3 DynamoDB example using the existing ``first-table`` table.

Run from the project virtual environment:
    python dynamodb.py

The demo writes one temporary item, reads and updates it, then deletes it.
"""

import json
import os

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from dotenv import load_dotenv

TABLE_NAME = "first-table"
DEMO_ID = "dynamodb-demo-item"


def aws_session() -> boto3.Session:
    """Use AWS_REGION and AWS_PROFILE from .env, just like the Bedrock app."""
    load_dotenv()
    return boto3.Session(
        profile_name=os.getenv("AWS_PROFILE") or None,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def show(value: object) -> None:
    """Print DynamoDB responses in a readable format."""
    print(json.dumps(value, indent=2, default=str))


def run_demo() -> None:
    session = aws_session()
    dynamodb = session.resource("dynamodb")
    client = session.client("dynamodb")
    table = dynamodb.Table(TABLE_NAME)

    # list_tables: list table names available to this AWS identity.
    print("Available tables:")
    show(client.list_tables()["TableNames"])

    # describe_table: read the table schema and status.
    print("\nTable status:")
    show(client.describe_table(TableName=TABLE_NAME)["Table"]["TableStatus"])

    # put_item: create or replace an item with this partition-key value.
    table.put_item(
        Item={"id": DEMO_ID, "message": "Hello DynamoDB", "step": 1}
    )

    # get_item: fetch exactly one item by its primary key.
    print("\nItem after put_item:")
    show(table.get_item(Key={"id": DEMO_ID}).get("Item"))

    # update_item: change selected attributes without replacing the whole item.
    updated = table.update_item(
        Key={"id": DEMO_ID},
        UpdateExpression="SET #message = :message, #step = :step",
        ExpressionAttributeNames={"#message": "message", "#step": "step"},
        ExpressionAttributeValues={":message": "Updated item", ":step": 2},
        ReturnValues="ALL_NEW",
    )
    print("\nItem after update_item:")
    show(updated["Attributes"])

    # query: efficiently finds items with a specific partition-key value.
    # first-table has only the id partition key, so this returns at most one item.
    print("\nquery result:")
    show(table.query(KeyConditionExpression=Key("id").eq(DEMO_ID))["Items"])

    # scan: reads items across the table. Fine for a small demo, but avoid it
    # for routine lookups on large tables; use get_item or query instead.
    print("\nscan result (up to 10 items):")
    show(table.scan(Limit=10)["Items"])

    # delete_item: remove the temporary demo item by primary key.
    table.delete_item(Key={"id": DEMO_ID})
    print("\nDeleted the temporary demo item.")


if __name__ == "__main__":
    try:
        run_demo()
    except ClientError as error:
        print(f"DynamoDB request failed: {error}")
