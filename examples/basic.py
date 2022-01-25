# ================= Import Packages ================= #
import tensorflow as tf
from tensorflow import keras

from src.methods.base import StaticDistillation

# ================= Create Networks ================= #

# Create the teacher
teacher = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(32, 32, 3))

# Create the student
student = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(32, 32, 3))

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)

# ================= Dataset ================= #

# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0

# ================= Dataset ================= #

# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("----------------------Train the Teacher Alone----------------------")
# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, epochs=5)
teacher.evaluate(x_test, y_test)

# ================= Distil to Student================= #

# Initialize and compile distiller
distiller = StaticDistillation(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

print("----------------------Distil Teacher to Student----------------------")
# Distill teacher to student
distiller.fit(x_train, y_train, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)

# ================= Train from Scratch for Comparison ================= #

# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("----------------------Train Student from Scratch----------------------")
# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)
