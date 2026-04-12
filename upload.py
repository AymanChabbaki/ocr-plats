import os
from lightning_sdk import Studio

# Authentication
os.environ["LIGHTNING_USER_ID"] = "593bf4b8-324d-4d75-bbb0-3d9de49544bd"
os.environ["LIGHTNING_API_KEY"] = "c667f2ae-2b25-4e4e-93c0-e08d34ac9a5d"

studio = Studio(
    name="flying-jade-v3pp", 
    teamspace="vision-model", 
    user="aymanchabbaki-etu"
)

# Local path to your zip
local_zip = "C:/Users/HP ZBOOK/Desktop/New folder/yolo11s.pt"
# Remote path (it will land in /home/zeus/data.zip)
remote_zip = "yolo11s.pt" 

print("Uploading zip file... this is much faster!")
studio.upload_file(local_zip, remote_zip)
print("Upload complete!")