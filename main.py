import torch
import cv2
import time
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from utils.cam_utils import cam_init
from utils.train_utils import ResEmoteNetTrainer
from models.ResEmoteNet import ResEmoteNet
from data.FER2013_dataset import FER2013Dataset
from torchvision import transforms    
import torch.nn.functional as F
from PIL import Image


def train_model(model, lr=0.0015, optim=torch.optim.SGD, seed=42,
                loss_fn=torch.nn.CrossEntropyLoss, scheduler=None, 
                momentum=0.9, decay=1e-4,
                epochs=80, val_epoch=4, batch=16, track=True, 
                wandb_name='face-segmentation', save=True, PATH='./models/weights/weights.pth'):
    pl.seed_everything(seed)
    datamodule = FER2013Dataset(batch_size=batch)
    datamodule.setup()
    if track:
        wandb.init(project=wandb_name)
        logger = WandbLogger()

    net = model
    model = ResEmoteNetTrainer(model=net, lr=lr, optim=optim, sched=scheduler, loss=loss_fn, momentum=momentum, decay=decay)
    test_loader = datamodule.test_dataloader()

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=val_epoch,
    )

    # train
    trainer.fit(model, datamodule)
    trainer.test(model, test_loader)
    if track:
        wandb.finish()

    if save:
        torch.save(model.state_dict(), PATH)

def inference(model, img, cur, last_label, last_prob, last_time):
    if cur - last_time >= 2:
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Assuming the images are 48x48 pixels
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel images (RGB)
                transforms.ToTensor(),
            ])
        img = Image.fromarray(img)
        img = transform(img)
        img = img.unsqueeze(0)

        logits = model(img)
        out = F.softmax(logits, dim=1)
        idx = torch.argmax(out, dim=1)
        print(f"Inferring {out} {labels[idx.item()]}")
        cur = time.time()
        return labels[idx.item()], out[0][idx.item()], cur
    return last_label, last_prob, last_time


def main():
    cam, cascade, fps = cam_init()
    PATH = fr"./models/weights/weights_SGD.pth"
    kill_key = 13
    model = ResEmoteNet(inch=3, outch=7, softmax=False)
    model.load_state_dict(torch.load(PATH, weights_only=True), strict=False)    
    model.eval()
    exp = "Neutral"
    conf = 0
    last_time = time.time()

    while True:
        try:
            ret, frame = cam.read()
            
            if not ret:
                print("Error: Failed to grab frame.")
                cam.release()
                cv2.destroyAllWindows()
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cor = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            if len(face_cor) == 0:
                pass
            else:
                x1, y1, w, h = face_cor[0]
                face_image = frame[y1:y1+h, x1:x1+w]
                cur_time = time.time()
                exp, conf, last_time = inference(model, face_image, cur_time, exp, conf, last_time)
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 3)
                txt = f'Exp {exp} conf: {conf:.2f}'
                label = cv2.putText(frame, txt, (x1 + w + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Face Detection", frame)
            cv2.imshow("Face", face_image)

            if cv2.waitKey(10) == kill_key:
                cam.release()
                cv2.destroyAllWindows()
                break
        except KeyboardInterrupt:
            cam.release()
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    main()