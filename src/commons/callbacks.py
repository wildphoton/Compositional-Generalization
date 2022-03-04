#!/usr/bin/env python
"""
Created by zhenlinx on 04/13/2021
"""
from pytorch_lightning.callbacks import ModelCheckpoint

class MyModelCheckpoint(ModelCheckpoint):
    # def _get_metric_interpolated_filepath_name(
    #         self,
    #         ckpt_name_metrics: Dict[str, Any],
    #         epoch: int,
    #         step: int,
    #         trainer,
    #         del_filepath: Optional[str] = None,
    # ) -> str:
    #     filepath = self.format_checkpoint_name(epoch, step, ckpt_name_metrics)
    #     return filepath

    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)

        # version_cnt = self.STARTING_VERSION
        # while self.file_exists(filepath, trainer) and filepath != del_filepath:
        #     filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
        #     version_cnt += 1

        return filepath