import argparse
from processor import VideoProcessor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['webcam', 'video'], default='webcam')
    p.add_argument('--path', type=str, default=None)
    p.add_argument('--use_ml', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    vp = VideoProcessor(
        source=args.source,
        path=args.path,
        use_ml=args.use_ml,
        visualize=True
    )
    vp.run()

if __name__ == "__main__":
    main()
