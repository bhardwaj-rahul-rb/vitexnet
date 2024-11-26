import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import ViTexNetModel
from utils.dataset import QaTa


def main(args):
    print("CUDA available:", torch.cuda.is_available())
    
    ds_test = QaTa(csv_path=args.test_csv, root_path=args.test_root, tokenizer=args.bert_type, image_size=args.image_size, mode='test')
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=0)
    
    model = ViTexNetModel(args)
    
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    
    final = pl.Trainer(accelerator='gpu', devices=1)
    
    model.eval()
    
    print('Start testing')
    final.test(model, dataloaders=dl_test)
    print('Done testing')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViTexNet model testing")
    
    parser.add_argument("--test_csv", type=str, default="data/test_annotations.csv", help="Path to the test CSV file")
    parser.add_argument("--test_root", type=str, default="data/test", help="Root path to the test dataset")
    parser.add_argument("--bert_type", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized", help="Type of BERT tokenizer")
    parser.add_argument("--image_size", type=int, default=[224,224], help="Image size for input data")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation/testing")
    parser.add_argument("--checkpoint_path", type=str, default="results/vitexnet.ckpt", help="Path to the model checkpoint file")
    
    args = parser.parse_args()
    main(args)
