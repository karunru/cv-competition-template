# https://zenn.dev/link/comments/a863c94a4d4e1f

import datetime
import json
import os
import time
from datetime import timezone

import requests
from kaggle.api.kaggle_api_extended import KaggleApi


def slack_notify(msg="おわったよ"):
    proxies = {
        "http": "",
        "https": "",
    }

    slack_post_url = "https://slack.com/api/chat.postMessage"
    slack_token = os.environ["PET2_SLACK_TOKEN"]
    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer " + slack_token,
    }
    channel = "C02PSAJQYLS"
    data = {"channel": channel, "text": msg}
    return requests.post(
        slack_post_url, data=json.dumps(data), proxies=proxies, headers=headers
    )


api = KaggleApi()
api.authenticate()

COMPETITION = "petfinder-pawpularity-score"
result_ = api.competition_submissions(COMPETITION)[0]
latest_ref = str(result_)  # 最新のサブミット番号
submit_time = result_.date

status = ""

slack_notify(
    f"{result_.submittedBy}, {submit_time}, {result_.description}, {result_.url}"
)

while status != "complete":
    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if status == "complete":
        print("\r", f"run-time: {elapsed_time} min, LB: {result.publicScore}")
        slack_notify(
            f"run-time: {elapsed_time} min, LB: {result.publicScore}, {result_.url}"
        )
    else:
        print("\r", f"elapsed time: {elapsed_time} min", end="")
        time.sleep(60)
