from graph4nlp.pytorch.data.dataset import Text2LabelDataset, Text2LabelDataItem
from graph4nlp.pytorch.modules.utils.generic_utils import LabelModel
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from graph4nlp.pytorch.data.data import GraphData, to_batch
from copy import deepcopy
import torch
import pdb

class MovieDataset(Text2LabelDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'label'."""
        return {"vocab": "vocab.pt", "data": "data.pt", "label": "label.pt"}

    @staticmethod
    def collate_fn(data_list: [Text2LabelDataItem]):
        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)

        tgt = [float(deepcopy(item.output_label)) for item in data_list]
        tgt_tensor = torch.tensor(tgt).float()

        return {"graph_data": graph_data, "tgt_tensor": tgt_tensor}

    def download(self):
        # raise NotImplementedError(
        #     "This dataset is now under test and cannot be downloaded."
        #     "Please prepare the raw data yourself."
        #     )
        return

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format
        is specified by each individual task-specific base class. Returns
        all the indices of data items in this file w.r.t. the whole dataset.

        For Text2LabelDataset, the format of the input file should contain
        lines of input, each line representing one record of data. The
        input and output is separated by a tab(\t).

        Examples
        --------
        input: How far is it from Denver to Aspen ?    NUM

        DataItem: input_text="How far is it from Denver to Aspen ?", output_label="NUM"

        Parameters
        ----------
        file_path: str
            The path of the input file.

        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                index, output, input = line.split('\t')
                data_item = Text2LabelDataItem(
                    input_text=input.strip(), output_label=output.strip(), tokenizer=self.tokenizer
                )
                data.append(data_item)

        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        self.vocab_model = VocabModel.build(
            saved_vocab_file=self.processed_file_paths["vocab"],
            data_set=data_for_vocab,
            tokenizer=self.tokenizer,
            lower_case=self.lower_case,
            max_word_vocab_size=self.max_word_vocab_size,
            min_word_vocab_freq=self.min_word_vocab_freq,
            pretrained_word_emb_name=self.pretrained_word_emb_name,
            pretrained_word_emb_url=self.pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
            word_emb_size=self.word_emb_size,
            share_vocab=True,
        )

        # label encoding
        all_labels = {item.output_label for item in self.train + self.test}
        if "val" in self.__dict__:
            all_labels = all_labels.union({item.output_label for item in self.val})

        self.label_model = LabelModel.build(self.processed_file_paths["label"], all_labels=all_labels)

    def __init__(
        self,
        root_dir,
        topology_builder=None,
        topology_subdir=None,
        graph_type="static",
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        merge_strategy="tailhead",
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        word_emb_size=None,
        for_inference=None,
        reused_vocab_model=None,
        reused_label_model=None,
        **kwargs
    ):
        super(MovieDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_type=graph_type,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            max_word_vocab_size=max_word_vocab_size,
            min_word_vocab_freq=min_word_vocab_freq,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            word_emb_size=word_emb_size,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            reused_label_model=reused_label_model,
            **kwargs
        )
