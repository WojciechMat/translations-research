"""English-Polish parallel corpus extracted from Europarl-ST dataset."""

import datasets

_CITATION = """
@INPROCEEDINGS{
    jairsan2020a,
    author={
        J. {Iranzo-Sánchez}
        and J. A. {Silvestre-Cerdà}
        and J. {Jorge}
        and N. {Roselló}
        and A. {Giménez}
        and A. {Sanchis}
        and J. {Civera}
        and A. {Juan}
    },
    booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    title={Europarl-ST: A Multilingual Corpus for Speech Translation of Parliamentary Debates},
    year={2020},
    pages={8229-8233},
}
"""

_DESCRIPTION = """
English-Polish parallel corpus extracted from the Europarl-ST dataset.
This dataset contains only the text translations between English and Polish, ignoring the audio components
of the original dataset. The dataset is derived from tj-solergibert/Europarl-ST and
contains aligned translations from European Parliament speeches.
"""

_HOMEPAGE = "https://huggingface.co/datasets/tj-solergibert/Europarl-ST"

_LICENSE = "Creative Commons Attribution 4.0 International"

_ORIGINAL_DATASET = "tj-solergibert/Europarl-ST"


class EuroparlEnPlConfig(datasets.BuilderConfig):
    """BuilderConfig for English-Polish Europarl dataset."""

    def __init__(
        self,
        **kwargs,
    ):
        """BuilderConfig for English-Polish Europarl.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(EuroparlEnPlConfig, self).__init__(**kwargs)


class EuroparlEnPl(datasets.GeneratorBasedBuilder):
    """English-Polish parallel corpus from Europarl-ST dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        EuroparlEnPlConfig(
            name="en-pl",
            version=VERSION,
            description="English-Polish parallel corpus from Europarl-ST",
        ),
        EuroparlEnPlConfig(
            name="pl-en",
            version=VERSION,
            description="Polish-English parallel corpus from Europarl-ST",
        ),
    ]

    DEFAULT_CONFIG_NAME = "en-pl"

    def _info(self):
        if self.config.name == "en-pl":
            source_lang = "en"
            target_lang = "pl"
        else:
            source_lang = "pl"
            target_lang = "en"

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.Translation(
                        languages=(source_lang, target_lang),
                    ),
                    "segment_id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        # Load the original dataset
        original_dataset = datasets.load_dataset(
            _ORIGINAL_DATASET,
            trust_remote_code=True,
        )

        # Return split generators for each available split in the original dataset
        split_generators = []

        if "train" in original_dataset:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "split": "train",
                        "dataset": original_dataset["train"],
                    },
                )
            )

        if "valid" in original_dataset:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "split": "validation",
                        "dataset": original_dataset["valid"],
                    },
                )
            )

        if "test" in original_dataset:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "split": "test",
                        "dataset": original_dataset["test"],
                    },
                )
            )

        return split_generators

    def _generate_examples(self, split: str, dataset):
        """Yields examples."""
        if self.config.name == "en-pl":
            source_lang = "en"
            target_lang = "pl"
        else:
            source_lang = "pl"
            target_lang = "en"

        for idx, example in enumerate(dataset):
            # Get source and target text
            transcriptions = example["transcriptions"]

            # Skip if source or target language isn't available
            if source_lang not in transcriptions or target_lang not in transcriptions:
                continue

            # Skip if either translation is null
            if transcriptions[source_lang] is None or transcriptions[target_lang] is None:
                continue

            # Skip empty translations or those with just a period
            if transcriptions[source_lang].strip() in ["", "."] or transcriptions[target_lang].strip() in ["", "."]:
                continue

            # Create a unique segment ID
            original = example["original_language"]
            audio_path = example["audio_path"]
            segment_start = example["segment_start"]
            segment_end = example["segment_end"]
            segment_id = f"{original}/{audio_path}_{segment_start: .2f}_{segment_end: .2f}"

            yield idx, {
                "translation": {
                    source_lang: transcriptions[source_lang],
                    target_lang: transcriptions[target_lang],
                },
                "segment_id": segment_id,
            }
