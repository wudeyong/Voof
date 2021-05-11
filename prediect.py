import argparse

import torch

import preprocess_pwc
from cnn.NVIDIA import NVIDIA


def run(model_path, flow_eval_dataset):
    device = torch.device('cuda')
    model = NVIDIA(2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, (flow_stack, ) in enumerate(flow_eval_dataset):
        flow_stack = flow_stack.to(device)
        with torch.no_grad():
            prespeed = model(flow_stack)
            print("frame: " + str(i) + " => speed: " + prespeed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model path")
    parser.add_argument("--video_path", help="video path")
    args = parser.parse_args()
    video_path = args.video_path
    model_path = args.model_path

    dataset = preprocess_pwc.generate_optical_flow_dataset_pwc_by_video(video_path)
    run(model_path, dataset)
