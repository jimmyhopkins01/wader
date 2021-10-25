import os
import io
from io import BytesIO
import os.path
from google.cloud import vision_v1
from google.cloud import storage
import pandas as pd
import glob
import numpy as np
#ask user to uplpoad apikey.json here
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/input/f.filename'
client = vision_v1.ImageAnnotatorClient()
storage_client = storage.Client.from_service_account_json("/input/f.filename")

image_path = '/input/'+g.filename
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision_v1.types.Image(content=content)
response = client.label_detection(image=image)


#change this path to relative path rather than absolute path
save_path = '/output'
output_file='/output/'+g.filename+'.txt'

#files re uploaded from local folder name input. change it to the file uploaded by user

# temp = local_filenames[i]
# temp2 = temp.split("\\",-1)
# uploadfilename = "%s%s" % ('', temp2[1])    #here upload of what the user sends as image
# blob = bucket.blob(uploadfilename)
# blob.upload_from_filename(temp)

# input_image_uri = "gs://"+BUCKET_NAME+"/"+temp2[1]  #or just ask the user for link and feed it here
# output_file = os.path.join(save_path, temp2[1]+".txt")

# response = client.annotate_image({
# 'image': {'source': {'image_uri': input_image_uri}},
# 'features': [{"type_": vision_v1.Feature.Type.LABEL_DETECTION},
#                 {"type_": vision_v1.Feature.Type.WEB_DETECTION},
#                 {"type_": vision_v1.Feature.Type.OBJECT_LOCALIZATION},
#                 {"type_": vision_v1.Feature.Type.TEXT_DETECTION},
#             ]
# })
labels = response.label_annotations
# annotations = response.web_detection
# objects = response.localized_object_annotations
# texts = response.text_annotations

df = pd.DataFrame(columns=['description', 'best_guess_labels', 'web_entities','localized_objects','text_detection'])
for label in labels:
    df = df.append(
        dict(
            description=label.description
        ), ignore_index=True
    )
# if annotations.best_guess_labels:
#     for annotation in annotations.best_guess_labels:
#         df = df.append(
#             dict(
#             best_guess_labels = annotation.label
#             ), ignore_index=True
#         )
# if annotations.web_entities:
#     for entity in annotations.web_entities:
#         df = df.append(
#             dict(
#             web_entities = entity.description
#             ), ignore_index=True
#         )
# for object_ in objects:
#     df = df.append(
#             dict(
#             localized_objects = object_.name
#             ), ignore_index=True
#         )
# j=0
# for text in texts:
#     if j==0:
#         j=j+1
#         continue
#     df = df.append(
#             dict(
#             text_detection = text.description
#             ), ignore_index=True
#         )


#print these details rather than providing as an output. Maybe add option for both preview and download
df1 = df.replace(np.nan, '', regex=True)
with open(output_file, 'a+', encoding="utf-8") as f:
    dfAsString = df1.to_string(header=False, index=False)
    f.write(dfAsString)

