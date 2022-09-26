import datasets
import os

class TrustpilotConfig(datasets.BuilderConfig):
    """BuilderConfig for Trustpilot."""

    def __init__(
        self,
        factor,
        country,
        domain_train,
        domain_test,
        task,
        data_dir,
        **kwargs,
    ):
        """BuilderConfig for Trustpilot.
        Args:
          factor: `string`, which factor in use {gender, age}
          country: `string`, which country in use {denmark, germany, united_states, united_kingdom, france}
          domain_train: `string`, which domain in use {gender: F, M ; age: 0, 2} for training
          domain_test: `string`, which domain in use {gender: F, M ; age: 0, 2} for testing
          task: `string`, which task in use {rating, category, gender, age}
          data_dir: `string`, directory to load the file from
          **kwargs: keyword arguments forwarded to super.
        """
        super(TrustpilotConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.factor = factor
        self.country = country
        self.domain_train = domain_train
        self.domain_test = domain_test
        self.task = task
        self.data_dir = data_dir

_FACTOR = ["gender", "age"]
_COUNTRY = ["denmark", "germany", "united_states", "united_kingdom", "france"]
_DOMAIN_TRAIN = ["F", "M", "0", "2", "a_MIX", "g_MIX"]
_DOMAIN_TEST = ["F", "M", "0", "2", "a_MIX", "g_MIX"]
_TASK = ["rating", "category", "gender", "age", "class-age", "class-category", "class-rating", "class-gender"]

class Trustpilot(datasets.GeneratorBasedBuilder):
    """Trustpilot Dataset."""
    VERSION = datasets.Version("1.0.0")
    #BUILDER_CONFIG_CLASS = TrustpilotConfig
    BUILDER_CONFIGS = [
        TrustpilotConfig(
            factor=factor,
            country=country,
            domain_train=domain_train,
            domain_test=domain_test,
            task=task,
            name=factor+"."+country+"."+domain_train+"."+domain_test+"."+task,
            data_dir = f"./data/{factor}/{task}/",
            description=f"Factor: {factor}, Country: {country}, Domain_TRAIN: {domain_train}, Domain_TEST: {domain_test}, Task: {task}",
        )
        for factor in _FACTOR
        for country in _COUNTRY
        for domain_train in _DOMAIN_TRAIN
        for domain_test in _DOMAIN_TEST
        for task in _TASK
    ]

    def _info(self):
        print(self.config.data_dir)
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "review": datasets.Value("string"),
                    "label": datasets.Value("int8")
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )
    
    def _split_generators(self, dl_manager):
        data_file = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_file, ".".join([self.config.country, self.config.domain_train, "train"]))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_file, ".".join([self.config.country, self.config.domain_test, "test"]))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_file, ".".join([self.config.country, self.config.domain_train, "dev"]))},
            ),
                ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath,  "r") as f:
            reader = f.read().splitlines()   
        all_labels = sorted(set([txt.split('\t')[0] for txt in reader]))
        label_2_id = {k: v for v, k in enumerate(all_labels)}
        print(label_2_id)
        for id_, txt in enumerate(reader):
            label, review = txt.split('\t')
            label = label_2_id[label]
            yield id_, {"review": review, "label": label}