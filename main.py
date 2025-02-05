import torch
import cv2
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers  import WandbLogger
from pytorch_lightning.loggers  import WandbLogger
from utils.cam_utils import cam_init, get_face_img
from utils.train_utils import ResEmoteNetTrainer
from models.ResEmoteNet import ResEmoteNet
from data.FER2013_dataset import FER2013Dataset
from control import prep_model, inference


def train_model(model, lr=0.0015, optim=torch.optim.AdamW, 
                loss_fn=torch.nn.CrossEntropyLoss, scheduler=None, 
                epochs=800, val_epoch=20, batch=64, wandb=True, 
                wandb_name='awa', save=True, PATH='./models/weights/weights.pth'):
    datamodule = FER2013Dataset(batch_size=batch)
    datamodule.setup()
    if wandb:
        wandb.init(project=wandb_name)
        logger = WandbLogger()
    net = ResEmoteNet(inch=3, outch=7, softmax=False)
    model = ResEmoteNetTrainer(model=net, lr=lr, optim=optim, sched=scheduler, loss=loss_fn)
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
    if wandb:
        wandb.finish()

    if save:
        torch.save(model.state_dict(), PATH)

def main():
    cam, cascade, fps = cam_init()
    kill_key = 13
    net = ResEmoteNet(inch=3, outch=7, softmax=False)
    model = prep_model(net)

    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            cam.release()
            cv2.destroyAllWindows()
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cor = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(face_cor) == 0:
            pass
        else:
            x1, y1, w, h = face_cor[0]
            face_image = frame[y1:y1+h, x1:x1+w]
            cur_time = time.time()
            exp, conf, last_time = inference(model, face_image)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 3)
            txt = f'Expression {exp} with confidence of {conf:.2f}'
            label = cv2.putText(frame, txt, (x1 + w + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(10) == kill_key:
            cam.release()
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    pass