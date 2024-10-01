import os
import json
from tqdm import tqdm
import ipdb

jsonpath = "/mnt/bn/arnold-yy-audiodata/audioldm/metadata-master/processed/retrival_trainable/retrival_trainable_train_len50.json"

jsondata=[json.loads(line) for line in open(jsonpath, 'r')]

new_root = "/mnt/bn/lqhaoheliu/datasets/audiocaps/audios/train/"

jsonlen = len(jsondata)


# fw = open("/mnt/bn/arnold-yy-audiodata/audioldm/metadata-master/processed/retrival_trainable/retrival_trainable_train_len50.json", 'w', encoding='utf-8')
count=0
for i in tqdm(range(len(jsondata))):
    each = jsondata[i]
    wav = each["wav"]
    wav = new_root + wav[71:]

    if os.path.exists(wav):
        # label = each["label"]
        # caption = each["caption"]
        # score_list = each["score_list"]

        # new_writen = {
        #         "wav":wav,
        #         "label":label,
        #         "caption":caption,
        #         "score_list": score_list
        #         }

        # json.dump(new_writen, fw)
        # fw.write("\n")

        count+=1

print(f"the overall len is {jsonlen} and exit file num is {count}")




