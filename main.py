import os
import sys

from pano_process import generate_pano

def main():
    # ensure input and output in the same directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Files to be processed(shoule be put in the same directory)
    videos = ["1.mp4"]  # "2.mp4", "3.mp4"……

    for vid in videos:
        vid_path = os.path.join(base_dir, vid)
        if not os.path.isfile(vid_path):
            print(f"[Skip] failed to find this video：{vid_path}")
            continue

        name, _ = os.path.splitext(vid)
        out_name = f"pano_{name}.jpg"
        out_path = os.path.join(base_dir, out_name)

        print(f"[Start] Processing {vid} -> {out_name}")
        try:
            generate_pano(vid_path, out_path)
            print(f"[Done]")
        except Exception as e:
            print(f"[Error] When process {vid} :{e}\n")

if __name__ == "__main__":
    main()
