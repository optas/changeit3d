import torch
import numpy as np
from torch.utils.data import Dataset


class LanguageContrastiveDataset(Dataset):
    def __init__(self, data_frame, to_stimulus_func=None, shuffle_items=False, n_distractors=2,
                 shape_to_latent_code=None):
        """

        Args:
            data_frame:
            to_stimulus_func:
            shuffle_items:
            n_distractors:
            shape_to_latent_code:
        """
        super(LanguageContrastiveDataset, self).__init__()
        self.df = data_frame
        self.shuffle_items = shuffle_items
        self.n_distractors = n_distractors
        self.to_stimulus_func = to_stimulus_func
        self.shape_to_latent_code = shape_to_latent_code

    def __getitem__(self, index):
        row = self.df.iloc[index]
        tokens = row['tokens_encoded']
        tokens = np.array(tokens).T  # todo do via collate.

        item_ids = []
        for i in range(1, self.n_distractors + 1):
            item_ids.append(row[f'distractor_{i}'])

        item_ids.append(row['target'])  # now target is last.
        item_ids = np.array(item_ids)
        n_items = len(item_ids)
        label = n_items - 1

        if self.shuffle_items:
            idx = np.arange(n_items)
            np.random.shuffle(idx)
            item_ids = item_ids[idx]
            label = np.where(idx == label)[0][0]

        res = dict()
        res['tokens'] = tokens
        res['label'] = label
        res['index'] = index

        # load visual stimuli (images, point-clouds, pretrained vectors, whatever...)
        if self.to_stimulus_func is None:
            res['stimulus'] = index
        else:
            res['stimulus'] = []
            for x in item_ids:
                res['stimulus'].append(self.to_stimulus_func(x))
            res['stimulus'] = np.stack(res['stimulus'])
        return res

    def __len__(self):
        return len(self.df)


def sub_index_language_contrastive_dataloader(dataloader, indices, shuffle=False):
    """ Given a torch dataloader and a sequence of integers; extract the corresponding items of the
    carried dataset on the specific indices and make a new dataloader with them.

    Args:
        dataloader: torch.utils.data.DataLoader for LanguageContrastiveDataset
        indices: sequence of integers indexing the underlying dataset (rows of dataframe).
        shuffle: boolean, shuffle the order of the examples of the resulting dataloader
    Returns:
        new_loader: torch.utils.data.DataLoader for LanguageContrastiveDataset
    """

    dataset = dataloader.dataset
    sub_df = dataset.df.iloc[indices].copy(deep=True)
    sub_df.reset_index(inplace=True, drop=True)

    sub_dset = LanguageContrastiveDataset(data_frame=sub_df,
                                          to_stimulus_func=dataset.to_stimulus_func,
                                          shuffle_items=dataset.shuffle_items,
                                          n_distractors=dataset.n_distractors,
                                          shape_to_latent_code=dataset.shape_to_latent_code
                                          )

    batch_size = min(len(sub_df), dataloader.batch_size)

    new_loader = torch.utils.data.DataLoader(sub_dset,
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=dataloader.num_workers)
    return new_loader