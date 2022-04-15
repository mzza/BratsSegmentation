# uisng the pytorch lighting to  orgnize the training process
############
# 20200530
# 1. 修改每隔10次就保存一次检查点
# 2. 自动需寻找最好的lr
# 3. 训练500次，没有必要每次都验证
# 4. print learning rate delete loggers
# 5. 使用新的梯度下降算法
##############
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from UNet3D import UNet3D
from UNet3DGroup import UNet3DGroup
from UNet3DSpaceAttention import UNet3DSpaceAttention
from UNet3DSpaceAttentionFusion import UNet3DSpaceAttentionFusion
from UNet3DChannelAttention import UNet3DChannelAttention
from torch.optim import Adam, lr_scheduler
from LossFunction import DeepSupervisedCrossEntropyGeneralizedDiceLoss, DiceForSample, CrossEntropyGeneralizedDiceLoss
from torchvision import transforms
from torch.utils.data import DataLoader
from DataSet import BrainTumorDataSet
from Transforms import SpacialTransform, BrightnessTransform, ContrastTransform, ToTensor, MirrorTransform
from ParametersSetting import output_root_path, init_lr, max_epoch, training_result_folder
from pytorch_lightning.callbacks import ModelCheckpoint
from ParametersSetting import batch_size, patch_size, val_batch_size
from pytorch_lightning.loggers import TensorBoardLogger


class BrainTumorSeg(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.model = UNet3DGroup()
        self.root_path = hp["predict_root_path"]
        self.fold = hp["fold"]
        self.patch_size = hp["patch_size"]
        self.moving_average_dice = None
        self.batch_size = hp["batch_size"]
        self.val_batch_size = hp["val_batch_size"]

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        fold_excel_name = os.path.join(self.root_path, "KFold", "fold_" + str(self.fold) + ".xlsx")
        with pd.ExcelFile(fold_excel_name) as reader:
            self._train_data_path_list = list(pd.read_excel(reader, sheet_name="train")["predict_root_path"].values)
            self._val_data_path_list = list(pd.read_excel(reader, sheet_name="val")["predict_root_path"].values)
            # shape_List = pd.read_excel(reader, sheet_name="val")["current_shape"].values
        # shape_numpy = np.array([get_list_from_csv_string(shape_List[i]) for i in range(len(shape_List))])
        # max_shape = shape_numpy.max(axis=0, keepdims=False)
        # print("max_shape:" + str(max_shape))
        # self._val_shape = get_pad_shape(max_shape)
        # print("pad_shape:" + str(self._val_shape))

        print("num of train: " + str(len(self._train_data_path_list)))
        print("num of val: " + str(len(self._val_data_path_list)))

    def train_dataloader(self):
        trans = transforms.Compose([SpacialTransform(patch_size=self.patch_size),
                                    BrightnessTransform(sigma=0.1), ContrastTransform(range=(0.9, 1.1)),
                                    MirrorTransform(),
                                    ToTensor()])
        dataset = BrainTumorDataSet(image_path_list=self._train_data_path_list, transform=trans)
        dataloader_train = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        print("number of iteration per epoch on training " + str(len(dataloader_train)))
        return dataloader_train

    def configure_optimizers(self):
        self.opt = Adam(self.parameters(), lr=init_lr, weight_decay=1e-5)

        # sch_pla = lr_scheduler.ReduceLROnPlateau(optimizer=self.opt,
        #                                      mode='max', factor=0.2, patience=5,
        #                                      verbose=True, threshold=mini_delta,
        #                                      threshold_mode="abs")
        sch_lam = lr_scheduler.LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: 0.985 ** (epoch+1))

        # sch_pla_dict = {'scheduler': sch_pla,
        #        'interval': 'epoch',
        #             "monitor": "moving_average_dice",
        #        "frequency": 1}  # called after each training step

        return [self.opt], [sch_lam]

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        output = self.forward(x)
        loss = DeepSupervisedCrossEntropyGeneralizedDiceLoss(CrossEntropyGeneralizedDiceLoss())(output, y)
        #self.logger.experiment.add_scalar("train_loss", loss, global_step=self.global_step)
        return {'loss': loss}

    #     def training_epoch_end(self,attention_map_outputs):
    #         print("lr", self.opt.param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        output = self.forward(x)
        # metrics = DiceForSample()
        # metrics.forward(output, y)
        # val_loss = metrics.get_dice_loss()
        # dice_per_class = metrics.get_dice_per_class()
        # dice_region = metrics.get_dice_region()
        # return {'dice_perclass_batch': dice_per_class, "val_loss_batch": val_loss, "dice_region_batch": dice_region}
        return {"predict": output[-1].detach().cpu(), "label": y.detach().cpu()}

    def validation_epoch_end(self, outputs):
        epoch_output = torch.cat([x["predict"] for x in outputs], dim=0)
        epoch_label = torch.cat([x["label"] for x in outputs], dim=0)
        metrics = DiceForSample()
        metrics.forward(epoch_output, epoch_label)
        # val_loss = metrics.get_dice_loss()
        val_dice_perclass = metrics.get_dice_per_class()
        val_dice_region = metrics.get_dice_region()
        print("whole:" + str(val_dice_region[0]))
        print("core:" + str(val_dice_region[1]))
        print("enhance:" + str(val_dice_perclass[3]))
        print("edema:" + str(val_dice_perclass[2]))
        print("no_enhance:" + str(val_dice_perclass[1]))

        # self.logger.experiment.add_scalar("non_enhance", val_dice_perclass[1],global_step=self.global_step)
        # self.logger.experiment.add_scalar("edema", val_dice_perclass[2],global_step=self.global_step)
        # self.logger.experiment.add_scalar("enhance", val_dice_perclass[3],global_step=self.global_step)
        # self.logger.experiment.add_scalar("healthy", val_dice_perclass[0],global_step=self.global_step)
        #
        # self.logger.experiment.add_scalar("whole", val_dice_region[0],global_step=self.global_step)
        # self.logger.experiment.add_scalar("edema", val_dice_region[1],global_step=self.global_step)
        val_dice = val_dice_perclass[1:].mean()
        if self.current_epoch == 0:
            self.moving_average_dice = val_dice
        else:
            self.moving_average_dice = self.moving_average_dice * 0.9 + val_dice * 0.1
        #         self.logger.experiment.add_scalar("moving_average_loss", self.moving_average_dice)
        print("average performance:" + str(self.moving_average_dice))

        print("lr", self.opt.param_groups[0]['lr'])

        return {"moving_average_dice": self.moving_average_dice}

    def val_dataloader(self):
        dataset = BrainTumorDataSet(image_path_list=self._val_data_path_list,
                                    transform=transforms.Compose(
                                        [SpacialTransform(patch_size=self.patch_size), ToTensor()]))
        dataloader_val = DataLoader(dataset=dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=12)
        print("number of iteration per epoch on evauation " + str(len(dataloader_val)))
        return dataloader_val


def run_training(checkpoint=False):
    hp = {"batch_size": batch_size,
               "patch_size": patch_size,
               "predict_root_path": output_root_path,
               "fold": 1,
               "val_batch_size": val_batch_size}
    model = BrainTumorSeg(hp)

    results_save_path = os.path.join(output_root_path, training_result_folder)
    if not os.path.isdir(results_save_path):
        os.makedirs(results_save_path)
    num_gpus = torch.cuda.device_count()
    print("using gpus " + str(num_gpus))

    # stop_callback = EarlyStopping(monitor="moving_average_dice", min_delta=mini_delta, patience=200, verbose=True,mode="max")

    # logger_save_path = os.predict_root_path.join(output_root_path, "logger")
    # if not os.predict_root_path.isdir(logger_save_path):
    #     os.makedirs(logger_save_path)
    # logger = TensorBoardLogger(logger_save_path, name=logger_name)

    checkpoint_callback = ModelCheckpoint(filepath=results_save_path,
                                          monitor="moving_average_dice",
                                          verbose=True, save_top_k=-1, mode="max", period=20,
                                          save_weights_only=False)
    # lr_logger = LearningRateLogger()

    if checkpoint:
        print("train from chaeckpoint....")
        checkpoint_path = os.path.join(results_save_path, "epoch=0.ckpt")
        trainer = Trainer(resume_from_checkpoint=checkpoint_path,
                          gpus=num_gpus,
                          checkpoint_callback=checkpoint_callback,
                          check_val_every_n_epoch=20,
                          max_epochs=max_epoch,
                          train_percent_check=1, val_percent_check=1,
                          num_sanity_val_steps=2)
        trainer.fit(model)
    else:
        print("train from scrach...")
        trainer = Trainer(gpus=num_gpus,
                          early_stop_callback=None,
                          logger=None,
                          checkpoint_callback=checkpoint_callback,
                          check_val_every_n_epoch=20,
                          max_epochs=max_epoch,
                          train_percent_check=1, val_percent_check=1,
                          num_sanity_val_steps=1)
        # Run learning rate finder

        #         lr_finder = trainer.lr_find(model)
        #         print("mu")
        #         lr_finder.results

        #         print("sugesstion")
        #         new_lr = lr_finder.suggestion()
        #         print(new_lr)
        #         fig = lr_finder.plot(suggest=True)
        #         fig.savefig("1.png")
        trainer.fit(model)

if __name__ == '__main__':
    run_training(False)
