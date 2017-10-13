from tensorflow.contrib.data.python.ops import dataset_with_schema
from tensorflow.python.platform import test
import tensorflow as tf
import numpy as np


Circle = dataset_with_schema.build_dataset_element_type(
    'Circle', ['position', 'radius'],
    [[2],[]],
    [tf.float64, tf.float64]
)

Rectangle = dataset_with_schema.build_dataset_element_type(
    'Rectangle', ['position', 'dimension'],
    [[2],[2]],
    [tf.int64, tf.int64]
)



class RectanglesDatasetGenerator():

    @staticmethod
    def generate(samples):
        def _generator():
            for _ in range(samples):
                yield Rectangle(np.random.randint(0,5,2), np.random.randint(1,3,2))

        return dataset_with_schema.DatasetWithSchema.from_generator(_generator, Rectangle)


class DatasetWithSchemaTest(test.TestCase):


    def testSimple(self):
        dataset = RectanglesDatasetGenerator.generate(10)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        with self.test_session() as sess:
            for _ in range(10):
                result = sess.run(get_next)
                self.assertEqual(result.dimension.shape,(2,))
                self.assertEqual(result.position.shape, (2,))

    def testMap(self):
        dataset = RectanglesDatasetGenerator.generate(10)

        def inner_circle(rectangle):
            halved_dim = tf.multiply(0.5, tf.to_float(rectangle.dimension))
            position = tf.add(tf.to_float(rectangle.position), halved_dim)
            radius = tf.reduce_min(halved_dim)
            return Circle(position, radius)

        dataset = dataset.map(inner_circle, Circle)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        with self.test_session() as sess:
            for _ in range(10):
                result = sess.run(get_next)
                self.assertEqual(result.radius.shape,())

    def testBatch(self):
        dataset = RectanglesDatasetGenerator.generate(10).batch(5)

        #Batching an already batched dataset should fail
        with self.assertRaises(TypeError):
            dataset.batch(5)

        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        get_next = iterator.get_next()
        with self.test_session() as sess:
            sess.run(init_op)
            for _ in range(2):
                result = sess.run(get_next)
                # Should have the appropriate shape
                self.assertEqual(result.position.shape, (5, 2))
                self.assertEqual(result.dimension.shape, (5, 2))

        dataset = dataset.unbatch()

        #Unbatching a unbatched dataset should fail
        with self.assertRaises(TypeError):
            dataset.unbatch()

        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        get_next = iterator.get_next()

        with self.test_session() as sess:
            sess.run(init_op)
            for _ in range(10):
                result = sess.run(get_next)
                # Should have the appropriate shape
                self.assertEqual(result.position.shape, (2,))
                self.assertEqual(result.dimension.shape, (2,))



if __name__ == "__main__":
  test.main()