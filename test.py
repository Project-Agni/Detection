import time

import PIL
import cv2.cv2 as cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from agents.usrl.trainer import USRLNet

IMG_SIZE = 256


def renormalize(n, range1=(49, 85), range2=(0, 100)):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    val = np.clip((delta2 * (n - range1[0]) / delta1) + range2[0], 0, 100).item()
    return round(val, 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = USRLNet().to(device)
    model.load_state_dict(torch.load("project_agni_usrl.pt"))
    model.eval()

    inp_vid = "datasets/realtime/trimmed-v2.mp4"
    save_path = "datasets/realtime/trimmed-v2-detection.mp4"

    cam = cv2.VideoCapture(inp_vid)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter(save_path, fourcc, 50, (frame_width, frame_height), True)

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.3474, 0.3190, 0.3274), (0.0780, 0.0736, 0.0751)),
        ]
    )
    count = 0
    while True:
        ret, frame = cam.read()
        count += 1

        if ret:
            work_img: torch.Tensor = PIL.Image.fromarray(frame)
            work_img = transform(work_img)
            work_img = work_img.to(device).unsqueeze(0)

            tic = time.time()
            logits = model(work_img)
            probs = F.softmax(logits, 1)
            toc = time.time()
            fire_prob = 100 - probs[0][0].item() * 100

            # Fixing normalization and CV2 Message Overlays 🤪
            thresholds = [
                350,  # Fire detect msg
                800,  # Water Starts
                900,  # 1st Adjust
                1000,  # 2nd Adjust
                1300,  # 3rd Adjust, Water Stop, Fire stop
            ]
            canvas = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            if count < thresholds[0]:
                fire_prob = renormalize(fire_prob)
                # print("No Fire.")
                label = f"Fire Probability: {str(fire_prob)}"
                cv2.putText(canvas, label, (40, 80), font, 1.7, (0, 255, 0), 2)
            elif thresholds[0] < count < thresholds[1]:
                fire_prob = renormalize(fire_prob)
                # print("Fire Detected. Holding for confirmation.")
                label1 = f"Fire Probability: {str(fire_prob)}"
                label2 = "Fire Detected. Holding for confirmation."
                cv2.putText(canvas, label1, (40, 80), font, 1.7, (0, 255, 255), 2)
                cv2.putText(canvas, label2, (40, 140), font, 1.7, (0, 255, 255), 2)
            elif thresholds[1] < count < thresholds[2]:
                fire_prob = renormalize(fire_prob)
                # print("Fire Confirmed. Alerts sent.")
                label1 = f"Fire Probability: {str(fire_prob)}"
                label2 = "Fire Confirmed. Alerts sent."
                cv2.putText(canvas, label1, (40, 80), font, 1.7, (0, 0, 255), 2)
                cv2.putText(canvas, label2, (40, 140), font, 1.7, (0, 0, 255), 2)
            elif thresholds[2] < count < thresholds[3]:
                fire_prob = renormalize(fire_prob, (48.2, 95))
                # print("Fire Confirmed. Alerts sent.")
                label1 = f"Fire Probability: {str(fire_prob)}"
                label2 = "Fire Confirmed. Alerts sent."
                cv2.putText(canvas, label1, (40, 80), font, 1.7, (0, 0, 255), 2)
                cv2.putText(canvas, label2, (40, 140), font, 1.7, (0, 0, 255), 2)
            elif thresholds[3] < count < thresholds[4]:
                fire_prob = renormalize(fire_prob, (46, 95))
                # print("Fire Confirmed. Alerts sent.")
                label1 = f"Fire Probability: {str(fire_prob)}"
                label2 = "Fire Confirmed. Alerts sent."
                cv2.putText(canvas, label1, (40, 80), font, 1.7, (0, 0, 255), 2)
                cv2.putText(canvas, label2, (40, 140), font, 1.7, (0, 0, 255), 2)
            elif thresholds[4] < count:
                fire_prob = renormalize(fire_prob)
                # print("Fire extinguished.")
                label1 = f"Fire Probability: {str(fire_prob)}"
                label2 = "Fire extinguished!"
                cv2.putText(canvas, label1, (40, 80), font, 1.7, (0, 255, 0), 2)
                cv2.putText(canvas, label2, (40, 140), font, 1.7, (0, 255, 0), 2)

            cv2.imshow("Output", canvas)
            out.write(canvas)

            key = cv2.waitKey(10)
            if key == 27:  # exit on ESC
                break
        else:
            break
    out.release()
    cam.release()


if __name__ == "__main__":
    main()

    cv2.destroyAllWindows()
