import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import ModelNet, ShapeNet, ScanObject



class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args, _trsf = True, join_thelast = False):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed, trsf= _trsf)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            if join_thelast:
                if offset< increment:
                    self._increments[-1] = self._increments[-1] + offset
                else:
                    self._increments.append(offset)
            else:
                self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_olddata(self, per_class_num):
        x, y = self._train_data, self._train_targets
        data_list = []
        label_list = []
        start_idx = 0
        for idx in range(0, self.nb_tasks):
            data_list_tempt = []
            label_list_tempt = []
            for id in range(start_idx, start_idx + self._increments[idx]):
                class_data, class_targets = self._select(x, y, low_range=id, high_range=id +1)
                val_indx = np.random.choice(len(class_data), per_class_num, replace=False)
                data_list_tempt.append(class_data[val_indx])
                label_list_tempt.append(class_targets[val_indx])
            data_list.append(np.concatenate(data_list_tempt))
            label_list.append(np.concatenate(label_list_tempt))
            start_idx = start_idx + self._increments[idx]
        return data_list, label_list

    def get_olddata_Dummy(self, old_data, task_id):
        if task_id > 0:
            data = np.concatenate(old_data[0][: task_id])
            targets = np.concatenate(old_data[1][: task_id])
            return  DummyDataset(data, targets, None, False)
        else:
            return  None

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, old_data = None, cur_task = 0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        try:
            if mode == "train":
                trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            elif mode == "flip":
                trsf = transforms.Compose(
                    [
                        *self._test_trsf,
                        transforms.RandomHorizontalFlip(p=1.0),
                        *self._common_trsf,
                    ]
                )
            elif mode == "test":
                trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            else:
                raise ValueError("Unknown mode {}.".format(mode))
        except Exception:
            trsf = None

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if old_data is not None: # and cur_task - 1 >= 0:
            old_points =  old_data[0][:cur_task + 1]
            old_targets = old_data[1][:cur_task + 1]
            data = np.concatenate((data, np.concatenate(old_points)))
            targets = np.concatenate((targets, np.concatenate(old_targets)))

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        try:
            if mode == "train":
                trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            elif mode == "test":
                trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            else:
                raise ValueError("Unknown mode {}.".format(mode))
        except Exception:
            trsf = None

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, trsf = True):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        if trsf:
            self._train_trsf = idata.train_trsf
            self._test_trsf = idata.test_trsf
            self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            if self.trsf is not None:
                image = self.trsf(pil_loader(self.images[idx]))
            else:
                image = pil_loader(self.images[idx])
        else:
            if self.trsf is not None:
                image = self.trsf(Image.fromarray(self.images[idx]))
            else:
                image = self.images[idx]
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "modelnet40":
        return ModelNet()
    elif name == "shapenet":
        return ShapeNet()
    elif name == "scanobject":
        return ScanObject()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
