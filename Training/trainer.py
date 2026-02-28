import pytorch_lightning as pl
import wandb
import pandas as pd
import os
import shutil
import numpy as np

from sklearn.model_selection import GroupKFold

from Training.data_module import SkeletonDataModule
from Pipeline.classifier import TCN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

PATH_DATA        = "Dataset/Data/Processed/cls_data.csv"
PATH_LOGS        = "Training/Logs"
PATH_CHECKPOINTS = "Training/Checkpoints"
PATH_RESULTS     = "Experiments"

PROJECT_NAME     = "SaberPredict"

BATCH_SIZE       = 64
MAX_EPOCHS       = 75
TUNED_LR         = 2.8840315031266063e-05
N_SPLITS         = 6

def main():
    os.makedirs(PATH_RESULTS, exist_ok=True)

    df = pd.read_csv(PATH_DATA)
    df["bout_id"] = df["file"].str.split("/").str[0]
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_metrics = []
    best_fold = None
    best_split = None
    best_acc = -1
    best_ckpt_path = None

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(df, groups=df["bout_id"])
    ):

        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        wandb_logger = WandbLogger(
            project=PROJECT_NAME,
            name=f"TCN_fold_{fold}",
            save_dir=PATH_LOGS
        )

        data = SkeletonDataModule(
            train_df=train_df,
            test_df=test_df,
            batch_size=BATCH_SIZE,
            num_workers=2,
        )
        data.setup()

        model = TCN(
            num_classes=data.num_classes,
            label_dict=data.label_dict,
            max_epochs=MAX_EPOCHS,
            lr=TUNED_LR,
            #class_weights=data.class_weights,
            class_weights=None
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(PATH_CHECKPOINTS, f"fold_{fold}"),
            filename="best",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model, data)

        test_results = trainer.test(model, data, ckpt_path="best")[0]

        test_acc = test_results["test_acc"]
        fold_metrics.append(test_acc)

        print(f"Fold {fold} Test Accuracy: {test_acc:.4f}")

        # Track best fold
        if test_acc > best_acc:
            best_acc = test_acc
            best_fold = fold
            best_split = test_df["bout_id"].unique()[0]
            best_ckpt_path = checkpoint_callback.best_model_path

        wandb.finish()

    # ===== Aggregate Results =====
    mean_acc = np.mean(fold_metrics)
    std_acc  = np.std(fold_metrics)

    print("\n===== Cross Validation Summary =====")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Accuracy : {std_acc:.4f}")
    print(f"Best Fold    : {best_fold} ({best_acc:.4f})")
    print(f"Best Split   : {best_split}")

    # Save metrics
    results_df = pd.DataFrame({
        "fold": list(range(N_SPLITS)),
        "accuracy": fold_metrics
    })

    results_df.loc["mean"] = ["-", mean_acc]
    results_df.loc["std"]  = ["-", std_acc]

    results_df.to_csv(os.path.join(PATH_RESULTS, "cv_metrics.csv"), index=False)

    # ===== Save Best Fold Model =====
    if best_ckpt_path is not None:
        final_model_path = os.path.join(PATH_RESULTS, "best_fold_model.ckpt")
        shutil.copy(best_ckpt_path, final_model_path)
        print(f"\nBest fold model saved to: {final_model_path}")

if __name__ == "__main__":
    main()