"""
USAGE
To read and write back out to video:
--input videos/sample_2.mp4 --output output/demo.avi -t 132,140

To read from webcam and write back out to disk:
python main.py --output output/webcam_output.avi

Run from terminal
python3 main.py --use-gpu 1 --input videos/sample_2.mp4 --output output/webcam_output.avi

"""
import argparse
import queue
from pathlib import Path

from payment_server import PaymentServerThread
from people_tracker import PeopleTracker


def main():
    # Get and parse the input arguments
    args = read_parse_args()

    # Thread-safe queue that stores payment signals
    payment_id_queue = queue.Queue()

    payment_server = PaymentServerThread(payment_id_queue)

    people_tracker = PeopleTracker(
        payment_id_queue,  # Thread safe queue for reading if payment event occurred
        args["terminals"],
        args["line_div"],
        args["pbtxt"],
        args["model"],
        args["use_gpu"],
        args.get("input", None),
        args.get("output", None)
    )
    payment_server.start()

    # This is run on the main thread, hence on termination code will continue
    people_tracker.run()

    # Wait for the server to close down
    payment_server.stop()
    payment_server.join()


def read_parse_args():
    cdir = Path(__file__).parent
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pbtxt", help="path to Coco pbtxt file",
                    default=str(cdir / "faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"))
    ap.add_argument("-m", "--model", help="path to Coco pre-trained model",
                    default=str(cdir/ "faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.pb"))
    ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-t", "--terminals", type=coords, nargs='+',
                    help="location of payment terminals", default=[(350, 200)])
    ap.add_argument("-u", "--use-gpu", type=bool, default=False,
                    help="boolean indicating if CUDA GPU should be used")
    ap.add_argument("-l", "--line_div", type=float, default=1.5)
    return vars(ap.parse_args())

def coords(arg_string):
    try:
        x, y = map(int, arg_string.split(','))
        return x, y
    except Exception as e:
        raise argparse.ArgumentTypeError("coords must be of the form 'x,y'") from e


if __name__ == "__main__":
    "If this file is run from the console as a run script"
    main()
