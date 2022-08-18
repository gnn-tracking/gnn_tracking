from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import partial

import mdmm
import numpy as np
import pandas as pd
import torch
from models.track_condensation_networks import PointCloudTCN
from torch import nn, optim

from gnn_tracking.utils.graph_datasets import get_dataloaders, initialize_logger
from gnn_tracking.utils.losses import (
    BackgroundLoss,
    EdgeWeightLoss,
    ObjectLoss,
    PotentialLoss,
)
from gnn_tracking.utils.training import binary_classification_stats

initialize_logger(verbose=False)


def parse_args(args):
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("-i", "--indir", type=str, default="graphs/train1_ptmin0")
    add_arg("-o", "--outdir", type=str, default="trained_models")
    add_arg("--stat-outfile", type=str, default="test")
    add_arg("--model-outfile", type=str, default="test")
    add_arg("--predict-track-params", action=argparse.BooleanOptionalAction)
    add_arg("-v", "--verbose", action=argparse.BooleanOptionalAction)
    add_arg("--n-train", type=int, default=120)
    add_arg("--n-test", type=int, default=30)
    add_arg("--n-val", type=int, default=10)
    add_arg("--learning-rate", type=float, default=10**-4)
    add_arg("--gamma", type=float, default=1)
    add_arg("--step-size", type=int, default=10)
    add_arg("--n-epochs", type=int, default=250)
    add_arg("--log-interval", type=int, default=10)
    add_arg("--q-min", type=float, default=0.001)
    add_arg("--sb", type=float, default=0.01)
    add_arg("--pt-min", type=float, default=0.0)
    add_arg("--loss-c-scale", type=float, default=1.0)
    add_arg("--loss-b-scale", type=float, default=1.0)
    add_arg("--loss-o-scale", type=float, default=1.0)
    add_arg("--save-models", type=bool, default=False)
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    return parser.parse_args(args)


class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters
        (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device,
        even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()

        if torch.cuda.is_available():
            self.noise_params = torch.rand(
                n_losses, requires_grad=True, device="cuda:0"
            )
        else:
            self.noise_params = torch.rand(n_losses, requires_grad=True)

    def forward(self, losses):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1 / torch.square(self.noise_params[i])) * loss + torch.log(
                self.noise_params[i]
            )

        return total_loss


class TCNTrainer:
    def __init__(self, args, model, loaders, device="cpu", constrain_loss=False):
        self.args = args
        self.constrain_loss = constrain_loss
        logging.info(f"Using arguments: {args}")
        self.model = model.to(device)
        self.device = device
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]
        self.edge_weight_loss = EdgeWeightLoss()
        self.potential_loss = PotentialLoss(q_min=args.q_min, device=device)
        self.background_loss = BackgroundLoss(
            q_min=args.q_min, device=device, sb=args.sb
        )
        self.object_loss = ObjectLoss(
            q_min=args.q_min, device=device, sb=args.sb, mode="purity"
        )

        # quantities to predict
        self.W = torch.empty(1, dtype=torch.float, device=device)  # edge weights
        self.B = torch.empty(
            1, dtype=torch.float, device=device  # condensation likelihoods
        )
        self.H = torch.empty(
            1, dtype=torch.float, device=device  # clustering coordinates
        )
        self.Y = torch.empty(1, dtype=torch.float, device=device)  # edge truth labels
        self.L = torch.empty(1, dtype=torch.float, device=device)  # hit truth labels
        self.P = torch.empty(1, dtype=torch.float, device=device)  # track parameters

        # build a constrained optimizer
        self.multi_loss = MultiNoiseLoss(n_losses=4).to(device)
        self.optimizer = optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.multi_loss.noise_params},
            ],
            lr=args.learning_rate,
        )
        logging.info("Parameter groups:", self.optimizer.param_groups)

        def lambda1(epoch):
            return 1 / (2 ** ((epoch + 11) // 10))

        # noinspection PyUnusedVariable
        def lambda2(epoch):
            return 1

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lambda1, lambda2]
        )

        # output quantities
        self.train_loss = []
        self.test_loss = []

    def zero_divide(self, a, b):
        return a / b if (b != 0) else 0

    def build_constrained_optimizer(self):
        args = self.args
        loss_W = partial(self.edge_weight_loss, self.W, self.B, self.H, self.Y, self.L)
        loss_B = partial(self.background_loss, self.W, self.B, self.H, self.Y, self.L)
        constrain_W = mdmm.MaxConstraint(loss_W, 0.2)
        constrain_B = mdmm.MaxConstraint(loss_B, 0.97)
        mdmm_module = mdmm.MDMM([constrain_W, constrain_B])
        optimizer = mdmm_module.make_optimizer(
            [self.W, self.B, self.H, self.Y, self.L],
            lr=args.learning_rate,
            optimizer=optim.Adam,
        )
        self.mdmm_module = mdmm_module
        self.optimizer = optimizer

    def validate(self):
        args = self.args
        self.model.eval()
        opt_thlds, accs = [], []
        for batch_idx, (data, f) in enumerate(self.val_loader):
            data = data.to(self.device)
            if args.predict_track_params:
                self.W, self.H, self.B, self.P = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            else:
                self.W, self.H, self.B = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            self.Y, self.W = data.y, self.W.squeeze(1)
            self.L = data.particle_id
            diff, opt_thld, opt_acc = 100, 0, 0
            for thld in np.arange(0.01, 0.5, 0.001):
                acc, TPR, TNR = binary_classification_stats(self.W, self.Y, thld)
                delta = abs(TPR - TNR)
                if delta < diff:
                    diff, opt_thld, opt_acc = delta, thld, acc
            opt_thlds.append(opt_thld)
            accs.append(opt_acc)
        logging.info(f"Optimal edge weight threshold: {np.mean(opt_thlds):.5f}")
        return np.nanmean(opt_thlds)

    def test(self, epoch, thld=0.5):
        args = self.args
        self.model.eval()
        losses = {"total": [], "W": [], "V": [], "B": [], "P": [], "acc": []}
        with torch.no_grad():
            for batch_idx, (data, f) in enumerate(self.test_loader):
                data = data.to(self.device)
                if args.predict_track_params:
                    self.W, self.H, self.B, self.P = self.model(
                        data.x, data.edge_index, data.edge_attr
                    )
                else:
                    self.W, self.H, self.B = self.model(
                        data.x, data.edge_index, data.edge_attr
                    )
                self.L = data.particle_id
                self.Y, self.W = data.y, self.W.squeeze(1)
                self.L, self.T = data.particle_id, data.track_params
                self.R = data.reconstructable.long()
                loss_W = self.edge_weight_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                loss_V = self.potential_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                loss_B = self.background_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                acc, TPR, TNR = binary_classification_stats(self.W, self.Y, thld)

                losses["total"].append(loss_W + loss_V + loss_B)
                losses["W"].append(loss_W)
                losses["V"].append(loss_V)
                losses["B"].append(loss_B)
                losses["acc"].append(acc.item())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        logging.info(f"Total Test Loss: {losses['total']:.5f}")
        logging.info(f"Edge Classification Loss: {losses['W']:.5f}")
        logging.info(f"Edge Classification Accuracy: {losses['acc']:.5f}")
        logging.info(f"Condensation Test Loss: {losses['V']:.5f}")
        logging.info(f"Background Test Loss: {losses['B']:.5f}")
        self.test_loss.append(pd.DataFrame(losses, index=[epoch]))

    def train(self, epoch):
        args = self.args
        self.model.train()
        losses = {
            "total": [],
            "W": [],
            "V": [],
            "B": [],
            "P": [],
            "Wr": [],
            "Vr": [],
            "Br": [],
            "Pr": [],
        }
        for batch_idx, (data, f) in enumerate(self.train_loader):
            data = data.to(self.device)
            if args.predict_track_params:
                self.W, self.H, self.B, self.P = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
                self.T = data.track_params
            else:
                self.W, self.H, self.B = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            self.Y, self.W = data.y, self.W.squeeze(1)
            self.L, self.T = data.particle_id, data.track_params
            self.R = data.reconstructable.long()
            loss_W = self.edge_weight_loss(self.W, self.B, self.H, self.Y, self.L)
            loss_V = self.potential_loss(self.W, self.B, self.H, self.Y, self.L)
            loss_B = self.background_loss(self.W, self.B, self.H, self.Y, self.L)
            loss_P = self.object_loss(
                self.W, self.B, self.H, self.P, self.Y, self.L, self.T, self.R
            )
            loss = self.multi_loss([loss_W, loss_V, loss_B, loss_P])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not (batch_idx % 10):
                logging.info(
                    f"Epoch {epoch} ({batch_idx}/{len(self.train_loader)}):"
                    + f" loss={loss.item():.5f}; loss_W={loss_W.item():.5f};"
                    + f" loss_V={loss_V.item():.5f}; loss_B={loss_B.item():.5f};"
                    + f" loss_P={loss_P.item():.5f}"
                )
                logging.info(f"Track parameter predictions: \n{self.P[0:5]}")
                logging.info(f"True track parameters: \n{self.T[0:5]}")
                logging.info(f"Reconstructable: \n{self.R[0:5]}")
            losses["total"].append(loss_W.item() + loss_V.item() + loss_B.item())
            losses["W"].append(loss_W.item())
            losses["V"].append(loss_V.item())
            losses["B"].append(loss_B.item())
            losses["P"].append(loss_P.item())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        logging.info(f"Total Train Loss: {losses['total']:.5f}")
        logging.info(f"Edge Classification Loss: {losses['W']:.5f}")
        logging.info(f"Condensation Train Loss: {losses['V']:.5f}")
        logging.info(f"Background Train Loss: {losses['B']:.5f}")
        self.train_loss.append(pd.DataFrame(losses, index=[epoch]))

    def train_loop(self):
        args = self.args
        for epoch in range(1, args.n_epochs + 1):
            logging.info(f"---- Epoch {epoch} ----")
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()
            if args.save_models:
                model_out = os.path.join(
                    args.outdir, f"{args.model_outfile}_epoch{epoch}.pt"
                )
                torch.save(model.state_dict(), model_out)

            # if (self.args.save_models):
            #    model_out = os.path.join(self.args.outdir,
            #                             f"{self.args.model_outfile}_epoch{epoch}.pt")
            #    torch.save(model.state_dict(), model_out)
            #
            # stat_out = os.path.join(self.args.outdir,
            #                        f"{self.args.stat_outfile}.csv")
            # write_output_file(stat_out, self.args, pd.DataFrame(output))

    def write_output_file(self, fname, df={}):
        args = self.args
        f = open(fname, "w")
        f.write("# args used in training routine:\n")
        for arg in vars(args):
            f.write(f"# {arg}: {getattr(args,arg)}\n")
        df.to_csv(f)
        f.close()


def main(args):
    # initialize
    args = parse_args(args)
    logging.info(f"Using arguments {args}")
    initialize_logger(args.verbose)
    # job_name = f"TCN_mdmm_pt{args.pt_min}"

    # import dataloaders
    params = {"batch_size": 1, "shuffle": True, "num_workers": 4}
    loaders = get_dataloaders(
        args.indir, args.n_train, args.n_test, args.n_val, shuffle=False, params=params
    )

    # use cuda (gpu) if possible, otherwise fallback to cpu
    use_cuda = torch.cuda.is_available()
    logging.info(f"Parameter use_cuda={use_cuda}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Utilizing {device}")

    # instantiate instance of the track condensation network
    if args.predict_track_params is not None:
        predict_track_params = True
    model = PointCloudTCN(5, 4, 5, predict_track_params=predict_track_params).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total Trainable Params: {total_trainable_params}")

    # build a trainer
    trainer = TCNTrainer(args, model, loaders, device)
    trainer.train_loop()


if __name__ == "__main__":
    main(sys.argv[1:])
