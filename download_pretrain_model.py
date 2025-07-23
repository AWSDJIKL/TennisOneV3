import gdown
import tarfile
import os

download_url = [
    ["TrackNet_best.tar", "https://drive.google.com/file/d/1d66u5brtaM-GF3KkHV6dba6xB0y-44Jv/view?usp=sharing",
     ["TrackNetV3/exp"]],
    ["work_dir.tar", "https://drive.google.com/file/d/1jXLxYv7WS5GQqqIYit6itdClzunMkOmS/view?usp=sharing",
     ["TrackNetV3/GCN/work_dir"]],
]

for tar_file_name, url, unzip_dir in download_url:
    gdown.download(url, tar_file_name, quiet=False, fuzzy=True)
    for path in unzip_dir:
        with tarfile.open(tar_file_name, mode='r:*') as tar:
            tar.extractall(path=path)
            print(f"已解压 {tar_file_name} 到 {path}")
    os.remove(tar_file_name)
