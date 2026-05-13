"""Project-specific Lightning callbacks."""
from __future__ import annotations

import lightning as L


class ValSummaryCallback(L.Callback):
    """After every validation epoch, print an Ultralytics-style table to stdout.

    Reads `pl_module._last_val_metrics` populated by
    `DetLitModule.on_validation_epoch_end`.

    Format:
        [val epoch N]            Images   Instances    mAP50   mAP   mAP75   mAR100
        mot17_val_half             2659       45203    0.816  0.494  0.561   0.660
        crowdhuman_val             4370       99001    0.764  0.463  0.535   0.690
    """

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metrics = getattr(pl_module, "_last_val_metrics", None)
        if not metrics:
            return
        ep = trainer.current_epoch
        # Sample counts per val set — look them up from the LightningDataModule
        dm = trainer.datamodule
        try:
            n_images = {name: len(dm.val_sets[name]) for name in metrics.keys()}
        except AttributeError:
            n_images = {name: -1 for name in metrics.keys()}

        header = f"\n[val epoch {ep:>3}]   {'Images':>10}  {'mAP50':>7}  {'mAP':>7}  {'mAP75':>7}  {'mAR100':>7}"
        rows = []
        for name, sub in metrics.items():
            rows.append(
                f"{name:<22}{n_images.get(name, -1):>10}  "
                f"{sub.get('mAP50', 0):>7.3f}  "
                f"{sub.get('mAP',   0):>7.3f}  "
                f"{sub.get('mAP75', 0):>7.3f}  "
                f"{sub.get('mAR100', 0):>7.3f}"
            )
        # Use trainer.print to honor DDP rank (only prints on rank 0)
        trainer.print(header)
        for r in rows:
            trainer.print(r)
        trainer.print("")  # blank line


__all__ = ["ValSummaryCallback"]
