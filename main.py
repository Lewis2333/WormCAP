import argparse
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from core.tracker import VideoTracker
from utils import visualization, io_utils


def main():
    parser = argparse.ArgumentParser(description="C. elegans Tracking System")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to PaddleX inference model directory')
    parser.add_argument('--model_name', type=str, default="Mask-RT-DETR-H", help='Model architecture name')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory to save results')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show processing window (slow)')
    parser.add_argument('--save_video', action='store_true', help='Save visualized video')

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return

    print("=" * 60)
    print(f"Initializing Tracker")
    print(f"Model: {args.model_dir}")
    print(f"Video: {args.video}")
    print("=" * 60)

    tracker = VideoTracker(
        model_name=args.model_name,
        model_dir=args.model_dir,
        conf_threshold=args.conf
    )

    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save_video:
        out_path = Path(args.output_dir) / "tracked_video.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Processing")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            tracks = tracker.update(frame, frame_id, fps)

            if args.show or args.save_video:
                vis_frame = visualization.draw_tracks(frame, tracks, frame_id)
                if args.save_video:
                    writer.write(vis_frame)
                if args.show:
                    cv2.imshow('Tracker', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            pbar.update(1)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        pbar.close()

    print("\nExporting data...")
    io_utils.export_data(tracker.track_history, tracker.track_features, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()