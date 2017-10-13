from tensorflow.contrib.data.python.ops import dataset_ops as contrib_dataset_ops
from collections import namedtuple
import tensorflow as tf


class DatasetElementMeta(type):
    shapes = None
    types = None


def build_dataset_element_type(typename, field_names, _shapes, _types):
    schema = namedtuple(typename, field_names)

    class DatasetElement(schema, metaclass=DatasetElementMeta):
        shapes = schema(*_shapes)
        types = schema(*_types)
    return DatasetElement


class BatchedSchemaMeta(DatasetElementMeta):
    pass


def batchify_schema(Schema):
    #TODO: handle nested structures
    assert(min([ss is None or isinstance(ss, int) for s in Schema.shapes for ss in s]))
    if isinstance(Schema, BatchedSchemaMeta):
        raise TypeError("The schema is already of type ", BatchedSchemaMeta)
    assert(not isinstance(Schema, BatchedSchemaMeta))
    class BatchedSchema(Schema, metaclass=BatchedSchemaMeta):
        shapes = Schema(*[[None] + s for s in Schema.shapes])
    return BatchedSchema

def unbatchify_schema(Schema):
    assert(isinstance(Schema, BatchedSchemaMeta))
    return Schema.__bases__[0]


class DatasetWithSchema(contrib_dataset_ops.Dataset):

    def __init__(self, dataset, input_schema, output_schema):
        if dataset is not None:
            super(DatasetWithSchema, self).__init__(dataset)
        self.input_schema = input_schema
        self.output_schema = output_schema

    def _apply_op(self, operator, output_schema, args=[], kwargs={}):
        dataset = DatasetWithSchema(None, self.output_schema, output_schema)
        super(DatasetWithSchema, dataset).__init__(
            getattr(super(DatasetWithSchema, self), operator)(*args, **kwargs)
        )
        return dataset

    def batch(self, batch_size):
        if self.is_batched:
            raise TypeError("The dataset is already batched")
        return self._apply_op('batch',  batchify_schema(self.output_schema), args=[batch_size])

    def unbatch(self):
        if not self.is_batched:
            raise TypeError("The dataset is not batched")
        return self._apply_op('apply',  unbatchify_schema(self.output_schema), args=[tf.contrib.data.unbatch()])

    @property
    def is_batched(self):
        return isinstance(self.output_schema, BatchedSchemaMeta)

    def map(self,
            map_func,
            output_schema,
            num_parallel_calls=None):
        def _map_func(input):
                input_with_schema = self.output_schema(*input)
                return map_func(input_with_schema)
        return self._apply_op('map', output_schema, args=[_map_func], kwargs={'num_parallel_calls': num_parallel_calls})

    @staticmethod
    def from_generator(generator, dataset_element_type):
        if not isinstance(dataset_element_type, DatasetElementMeta):
            raise TypeError(
                "dataset_element_type should be an instance of DatasetElementMeta, but is an instance of %s" % (
                    type(dataset_element_type)))

        return DatasetWithSchema(
            dataset=tf.data.Dataset.from_generator(
                generator,
                output_types=dataset_element_type.types,
                output_shapes=dataset_element_type.shapes
            ),
            input_schema=None,
            output_schema=dataset_element_type)



