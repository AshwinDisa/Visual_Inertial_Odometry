import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import TrajectoryDataModule
from models.bidirectional_lstm import BidirectionalLSTM

def main(args):
    # Create a directory for trained weights if it doesn't exist
    model_dir = f"trained_weights/{args.mode}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize the Data Module
    data_module = TrajectoryDataModule(json_file=args.data_file, batch_size=args.batch_size, mode=args.mode)
    print(len(data_module.train_dataloader().dataset))

    # Select the model based on the mode
    if args.mode == 'IO':
        model = BidirectionalLSTM(input_dim=6, hidden_dim=256, output_dim=7, num_layers=2)
    else:
        raise ValueError("Unsupported mode! Use 'IO' for inertial odometry.")

    # Logger setup
    logger = TensorBoardLogger("tb_logs", name=f"my_model_{args.mode}")

    # Checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,  # saves only the best model
        monitor='val_loss',  # metric to monitor
        mode='min',  # mode of the monitored quantity for optimization
        auto_insert_metric_name=False  # prevent automatic insertion of the metric name in the filename
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16 if args.fp16 else 32
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for different modalities")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--mode', type=str, choices=['VO', 'IO', 'VIO'], required=True, help='Mode of operation: VO, IO, or VIO')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')

    args = parser.parse_args()
    print(args)
    main(args)
