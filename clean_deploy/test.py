from get_data import *
from ml_utils import *


page = download_search_page("kaggle",1)
parsed = parse_search_page(page)

for i in parsed:
    video_page = download_video_page(i["link"])
    video_json_data = parse_video_page(video_page)
    #print(video_json_data.get('og:video:url', ''))
    #print(video_json_data['watch-title'])

    if 'watch-time-text' not in video_json_data:
        continue

    video_id = video_json_data.get('og:video:url', '')
    data_front = {"title": video_json_data['watch-title'], "video_id": video_id}

    print(compute_prediction(video_json_data))



