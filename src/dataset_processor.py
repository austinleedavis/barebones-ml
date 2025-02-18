from typing import Iterable, Mapping

from datasets import Dataset


class DatasetProcessor:
    """A class for applying a series of transformations to a Hugging Face `datasets.Dataset` object.

    Supported transformation types:
    - "select": Select specific indices from the dataset.
    - "select_columns": Select specific columns from the dataset.
    - "rename_columns": Rename columns from the dataset
    - "filter": Apply a filtering function to remove records based on conditions.
    - "map": Apply a transformation function to modify records in the dataset.

    Attributes:
        SUPPORTED_TRANSFORM_TYPES (list): A list of supported transformation types.
        transforms (Iterable[Mapping]): A list of transformation configurations.
    """

    SUPPORTED_TRANSFORM_TYPES = [
        FILTER,
        MAP,
        RENAME_COLUMNS,
        SELECT_COLUMNS,
        SELECT,
        TRAIN_TEST_SPLIT,
    ] = [
        "filter",
        "map",
        "rename_columns",
        "select_columns",
        "select",
        "train_test_split",
    ]

    transforms: Iterable[Mapping]

    def __init__(self, transforms: Iterable[Mapping]):
        """Initialize the DatasetProcessor with a list of transformation configurations.

        Args:
            transforms (Iterable[Mapping]): A list of dictionaries where each dictionary
                represents a transformation operation with a `type`, an optional `callable`,
                and optional `kwargs`.

        Raises:
            TypeError: If `transforms` is not an iterable of mappings.
            ValueError: If a transform has an invalid `type` or is missing required fields.
        """
        if not isinstance(transforms, Iterable):
            raise TypeError("Transforms must be an iterable of mappings.")

        for transform in transforms:
            if not isinstance(transform, Mapping):
                raise TypeError("Each transform must be a mapping.")

            t_type = transform.get("type")
            if t_type not in self.SUPPORTED_TRANSFORM_TYPES:
                raise ValueError(f"Invalid transform type: {t_type}")

            if t_type in {self.SELECT, self.FILTER, self.MAP} and "callable" not in transform:
                raise ValueError(f"Transform of type '{t_type}' requires a 'callable' field.")

            if (
                t_type in [self.SELECT_COLUMNS, self.RENAME_COLUMNS, self.TRAIN_TEST_SPLIT]
                and "kwargs" not in transform
            ):
                raise ValueError(
                    f"Transform of type '{t_type}' and requires a 'kwargs' field with 'column_names'."
                )

            if "kwargs" in transform and not isinstance(transform["kwargs"], Mapping):
                raise TypeError("Transform 'kwargs' must be a dictionary.")

        self.transforms = transforms

    def process(self, dataset: Dataset):
        """Apply the stored transformations to a given dataset.

        Args:
            dataset (Dataset): The Hugging Face dataset to be transformed.

        Returns:
            Dataset: The transformed dataset after applying all specified operations.

        Raises:
            ValueError: If an unknown transformation type is encountered.
        """
        for transform in self.transforms:
            transform_type = transform.get("type", None)
            kwargs = transform.get("kwargs", {})

            match transform_type:
                case self.SELECT:
                    indices = eval(transform.callable)
                    dataset = dataset.select(indices, **kwargs)
                case self.SELECT_COLUMNS:
                    dataset = dataset.select_columns(**kwargs)
                case self.RENAME_COLUMNS:
                    dataset = dataset.rename_columns(**kwargs)
                case self.FILTER:
                    filter_fn = eval(transform.callable)
                    dataset = dataset.filter(filter_fn, **kwargs)
                case self.MAP:
                    map_fn = eval(transform.callable)
                    dataset = dataset.map(map_fn, **kwargs)
                case self.TRAIN_TEST_SPLIT:
                    dataset = dataset.train_test_split(**kwargs)

        return dataset
