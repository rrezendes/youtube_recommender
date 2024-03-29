# deploy_front/app.py
#

import os.path
from flask import Flask
import os
import json
import run_backend

import time

app = Flask(__name__)


def get_predictions():
    videos = []

    novos_videos_json = "novos_videos.json"
    if not os.path.exists(novos_videos_json):
        run_backend.update_db()

    last_update = os.path.getmtime(novos_videos_json) * 1e9

    # if time.time_ns() - last_update > (720*3600*1e9): # aprox. 1 mes
    #    run_backend.update_db()

    with open("novos_videos.json", 'r') as data_file:
        for line in data_file:
            line_json = json.loads(line)
            videos.append(line_json)

    predictions = []
    for video in videos:
        # print(video)
        # print(video['video_id'])
        predictions.append((video['video_id'], video['title'], float(video['score'])))

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:30]

    predictions_formatted = []
    for e in predictions:
        # print(e)
        predictions_formatted.append(
            "<tr><th><a href=\"{link}\">{title}</a></th><th>{score}</th></tr>".format(title=e[1], link=e[0],
                                                                                      score=e[2]))

    return '\n'.join(predictions_formatted), last_update

@app.route('/')
def main_page():
    preds, last_update = get_predictions()
    return """<table>
    <tbody>
    <tr>
    <td>
    <h1>Youtube Video Recommender</h1>
    </td>
    </tr>
    <tr>
    <td>
    <p>Source code:&nbsp;</p>
    <p><a href="https://github.com/rrezendes/youtube_recommender">https://github.com/rrezendes/youtube_recommender</a></p>
    <p>Seconds from last update: {} </p>    
    </td>
    </tr>
    <tr>
    <p>&nbsp;</p>
    </tr>
    <tr>
    {}
    </tr>    
    </tbody>
    </table>""".format((time.time_ns() - last_update) / 1e9, preds)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
