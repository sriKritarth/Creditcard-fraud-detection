import os
import random

from dvclive import Live
import yaml

with Live() as live:
    epochs = yaml.safe_load(open("params.yaml"))["train_model"]["epochs"]
    live.log_param("epochs", epochs)
    for epoch in range(epochs):
        train_acc = epoch + random.random()
        train_loss = epochs - epoch - random.random()
        val_acc = epoch + random.random()
        val_loss = epochs - epoch - random.random()

        live.log_metric("train/accuracy", train_acc)
        live.log_metric("train/loss", train_loss)
        live.log_metric("val/accuracy", val_acc)
        live.log_metric("val/loss", val_loss)

with open("metrics.txt" , 'w') as outfile:
    outfile.write("Final val accuracy: " + str(val_acc) + "\n")
    outfile.write("Final loss accuracy: " + str(val_loss) + "\n")