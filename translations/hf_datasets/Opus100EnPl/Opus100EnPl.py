"""English-Polish parallel corpus extracted from Helsinki-NLP/opus-100 dataset."""

import datasets

_CITATION = """
@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta and Choukri, Khalid and Declerck, Thierry and Do{\u{g}}an, Mehmet U{\u{g}}ur and Maegaard, Bente and Mariani, Joseph and Moreno, Asuncion and Odijk, Jan and Piperidis, Stelios",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
}
"""

_DESCRIPTION = """
English-Polish parallel corpus extracted from the Helsinki-NLP/opus-100 dataset.
This dataset contains parallel translations between English and Polish from the OPUS collection.
OPUS-100 contains approximately 55M sentence pairs across 99 language pairs.
This subset focuses specifically on English-Polish translations with standardized formatting.
"""

_HOMEPAGE = "https://huggingface.co/datasets/Helsinki-NLP/opus-100"

_LICENSE = "Various open licenses (see original OPUS collection)"

_ORIGINAL_DATASET = "Helsinki-NLP/opus-100"


class Opus100EnPlConfig(datasets.BuilderConfig):
    """BuilderConfig for English-Polish OPUS-100 dataset."""

    def __init__(
        self,
        **kwargs,
    ):
        """BuilderConfig for English-Polish OPUS-100.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Opus100EnPlConfig, self).__init__(**kwargs)


class Opus100EnPl(datasets.GeneratorBasedBuilder):
    """English-Polish parallel corpus from OPUS-100 dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        Opus100EnPlConfig(
            name="en-pl",
            version=VERSION,
            description="English-Polish parallel corpus from OPUS-100",
        ),
        Opus100EnPlConfig(
            name="pl-en",
            version=VERSION,
            description="Polish-English parallel corpus from OPUS-100",
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
                    "source": datasets.Value("string"),  # Keep track of data source
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        # Load the original OPUS-100 dataset
        original_dataset = datasets.load_dataset(
            _ORIGINAL_DATASET,
            name="en-pl",  # OPUS-100 uses "en-pl" format
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

        if "validation" in original_dataset:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "split": "validation",
                        "dataset": original_dataset["validation"],
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
            # OPUS-100 format: {"translation": {"en": "text", "pl": "text"}}
            # or direct format: {"en": "text", "pl": "text"}

            if "translation" in example:
                # Standard translation format
                translation_data = example["translation"]
                source_text = translation_data[source_lang]
                target_text = translation_data[target_lang]
            else:
                # Direct format (fallback)
                source_text = example[source_lang]
                target_text = example[target_lang]

            # Skip if either text is None or empty
            if not source_text or not target_text:
                continue

            if source_text.strip() in ["", "."] or target_text.strip() in ["", "."]:
                continue

            yield idx, {
                "translation": {
                    source_lang: source_text,
                    target_lang: target_text,
                },
                "source": "opus-100",
            }
