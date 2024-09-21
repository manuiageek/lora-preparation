Some python scripts writing with the help of chatgpt that helps me prepare LoRA (Low Rank Adaptation) training for anime characters.
- 1_rename_mkv.py : it renames the titles of mkv episode to "01", "02" and so forth
- 2_ffmpeg_extr_jpg.py : extract frames from mkv with ffmpeg tool
- 3_countfiles_ssfolders.py : counting files from subfolders to check the export in jpg
- 4_detect_person_plus_delete.py : it loads yolov8x6_animeface model in order to check if AI can find a "face" inside loop of images in a folder.
Deleting any image that doesn't contain "face" in the frame.
