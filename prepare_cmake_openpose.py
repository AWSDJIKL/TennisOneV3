import gdown
import tarfile
import os

download_url = [
    ["3rdparty.tar", "https://drive.google.com/file/d/1jSpaNgoThNycs8gIWZEOBc846MXC6yX2/view?usp=sharing",
     ["./openpose"]],
    ["models.tar", "https://drive.google.com/file/d/1mAU5xbDm5HfRYcB1HIEKCgfXUvS_dJw9/view?usp=sharing",
     ["./openpose","TrackNetV3"]],
]

for tar_file_name, url, unzip_dir in download_url:
    gdown.download(url, tar_file_name, quiet=False, fuzzy=True)
    for path in unzip_dir:
        with tarfile.open(tar_file_name, mode='r:*') as tar:
            tar.extractall(path=path)
            print(f"已解压 {tar_file_name} 到 {path}")
    os.remove(tar_file_name)
