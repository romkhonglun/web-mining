import abc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd

# Import hàm tính metric cũ của bạn
from newsRecSys.model.naml.deeprec_utils import cal_metric


class DataIteratorAdapter(IterableDataset):
    """
    Wrapper để chuyển đổi custom iterator cũ của bạn thành PyTorch IterableDataset.
    Điều này giúp PyTorch Lightning có thể load data từ iterator cũ.
    """

    def __init__(self, iterator, news_file, behaviors_file):
        self.iterator = iterator
        self.news_file = news_file
        self.behaviors_file = behaviors_file

    def __iter__(self):
        """Yield từng batch data từ iterator gốc"""
        # Gọi load_data_from_file từ iterator gốc
        source = self.iterator.load_data_from_file(self.news_file, self.behaviors_file)
        for batch_data in source:
            yield batch_data


class BaseLightningModel(pl.LightningModule, abc.ABC):
    """
    Base class sử dụng PyTorch Lightning.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        super().__init__()
        self.save_hyperparameters(hparams)  # Tự động lưu hparams vào checkpoint
        self.iterator_creator = iterator_creator
        self.seed = seed

        if seed is not None:
            pl.seed_everything(seed)

        # Init các đường dẫn file data (cần được set trước khi fit)
        self.train_news_file = None
        self.train_behaviors_file = None
        self.valid_news_file = None
        self.valid_behaviors_file = None
        self.test_news_file = None
        self.test_behaviors_file = None

        # --- Build Model ---
        # Hàm này sẽ được implement ở class con để khởi tạo các layers
        self._build_model()

        # Init loss function
        self.loss_fn = self._get_loss()

    @abc.abstractmethod
    def _build_model(self):
        """Class con sẽ khởi tạo các layers (nn.Module) tại đây, gán vào self."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """
        Extract input và label từ batch dictionary.
        Returns:
            inputs: Tensor hoặc Dict of Tensors
            labels: Tensor
        """
        pass

    def forward(self, inputs):
        """
        Cần implement ở class con.
        PyTorch Lightning gọi hàm này khi inferenece.
        """
        raise NotImplementedError

    def _get_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif self.hparams.loss == "log_loss":
            return nn.BCELoss()  # Hoặc BCEWithLogitsLoss nếu output chưa qua sigmoid
        else:
            raise ValueError(f"Loss {self.hparams.loss} not defined")

    def configure_optimizers(self):
        """Thiết lập optimizer"""
        lr = self.hparams.learning_rate
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer not supported")
        return optimizer

    # ================= TRAINING LOOP =================
    def training_step(self, batch, batch_idx):
        """Xử lý 1 step training"""
        inputs, labels = self._get_input_label_from_iter(batch)

        # Forward pass (Lưu ý: inputs đã tự động được chuyển sang device)
        preds = self(inputs)

        # Xử lý output shape nếu cần (ví dụ model trả về tuple)
        if isinstance(preds, tuple): preds = preds[0]

        loss = self.loss_fn(preds, labels.float())

        # Logging (tự động hiện trên progress bar và lưu vào log)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ================= VALIDATION LOOP =================
    def validation_step(self, batch, batch_idx):
        """Xử lý 1 step validation"""
        inputs, labels = self._get_input_label_from_iter(batch)

        preds = self(inputs)
        if isinstance(preds, tuple): preds = preds[0]

        # Nếu dùng BCEWithLogitsLoss thì cần sigmoid ở đây để tính metric
        # Nếu dùng BCELoss thì preds đã là prob rồi
        # Giả sử cần sigmoid nếu output là logit:
        # preds = torch.sigmoid(preds) 

        loss = self.loss_fn(preds, labels.float())

        self.log("val_loss", loss, prog_bar=True)

        # Trả về data để dùng cho on_validation_epoch_end
        return {
            "preds": preds.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
            "impression_ids": batch["impression_index_batch"]  # Giả sử key này tồn tại
        }

    def on_validation_epoch_end(self):
        """
        Được gọi khi kết thúc 1 epoch validation.
        Tại đây ta gom toàn bộ preds lại để tính AUC, MRR, nDCG...
        """
        # Lấy outputs từ validation_step (PL lưu trữ sẵn trong validation_step_outputs nếu dùng list, 
        # nhưng ở bản mới cần override logic này hoặc dùng hook khác. 
        # Cách chuẩn hiện nay là tự lưu trữ trong một list self.val_outputs)
        pass
        # Note: Do PL thay đổi API phần này thường xuyên, 
        # cách an toàn nhất là implement custom logic gom data như dưới đây:

    # --- Custom Validation Aggregation ---
    # Do validation_step không tự động gom lại trong các phiên bản PL mới nhất theo cách cũ,
    # ta dùng biến tạm để lưu.
    def on_validation_start(self):
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        # ... code như trên ...
        inputs, labels = self._get_input_label_from_iter(batch)
        preds = self(inputs)
        if isinstance(preds, tuple): preds = preds[0]

        step_output = {
            "preds": preds.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
            "imp_ids": np.reshape(batch["impression_index_batch"], -1)
        }
        self.val_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self):
        # 1. Flatten all lists
        all_preds = np.concatenate([x['preds'].reshape(-1) for x in self.val_step_outputs])
        all_labels = np.concatenate([x['labels'].reshape(-1) for x in self.val_step_outputs])
        all_imp_ids = np.concatenate([x['imp_ids'] for x in self.val_step_outputs])

        # 2. Group labels (Logic cũ của bạn)
        group_impr_indexes, group_labels, group_preds = self.group_labels(
            all_labels, all_preds, all_imp_ids
        )

        # 3. Tính Metrics
        metrics = cal_metric(group_labels, group_preds, self.hparams.metrics)

        # 4. Log metrics
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=True, logger=True)

        # 5. Clear memory
        self.val_step_outputs.clear()

        print(f"\nValidation Metrics: {metrics}")

    # ================= DATA LOADERS =================
    def set_train_files(self, news_file, behaviors_file):
        self.train_news_file = news_file
        self.train_behaviors_file = behaviors_file

    def set_val_files(self, news_file, behaviors_file):
        self.valid_news_file = news_file
        self.valid_behaviors_file = behaviors_file

    def train_dataloader(self):
        iterator = self.iterator_creator(
            self.hparams, self.hparams.npratio, col_spliter="\t"
        )
        dataset = DataIteratorAdapter(iterator, self.train_news_file, self.train_behaviors_file)
        # batch_size=None vì iterator cũ của bạn đã tự batching rồi
        return DataLoader(dataset, batch_size=None)

    def val_dataloader(self):
        if not self.valid_news_file:
            return None
        iterator = self.iterator_creator(self.hparams, col_spliter="\t")
        dataset = DataIteratorAdapter(iterator, self.valid_news_file, self.valid_behaviors_file)
        return DataLoader(dataset, batch_size=None)

    # ================= HELPER METHODS =================
    def group_labels(self, labels, preds, group_keys):
        """Helper function giữ nguyên từ code cũ"""
        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds