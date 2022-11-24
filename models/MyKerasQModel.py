from ray.rllib.utils import try_import_tf
from .DistributionalQTFModel import OverrideDistributionalQTFModel
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel

tf = try_import_tf()[0]

#class MyKerasQModel(OverrideDistributionalQTFModel):
class MyKerasQModel(DistributionalQTFModel):
    def __init__(self, observation_space, action_space, num_outputs, model_config,
                 name="my_model",**kw):
        #import ipdb;ipdb.set_trace()
        # print("MyKerasQModel keys = {0}".format(model_config.keys()))
        # model_config["parameter_noise"] = kw["parameter_noise"]
        # del kw["parameter_noise"]
        # print(kw.items())
        # print("parameter noise, exists = ", ("parameter_noise" in kw.keys()))
        super(MyKerasQModel, self).__init__(
            observation_space, action_space, num_outputs, model_config, name, **kw)
        self.inputs = tf.keras.layers.Input(
            shape=observation_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            12,
            name="my_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.he_normal())(self.inputs)
        layer_2 = tf.keras.layers.Dense(
            8,
            name="my_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.he_normal())(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.softmax,
            kernel_initializer=tf.keras.initializers.he_normal())(layer_2)
        self.base_model = tf.keras.Model(self.inputs, layer_out)
        # print("num_outputs = ", num_outputs)
        # print("layer_1 = ", layer_1.shape)
        # print("layer_2 = ", layer_2.shape)
        # print("layer_out = ", layer_out.shape)
        # self.register_variables(self.base_model.variables)
    
    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    def metrics(self):
        return {"foo": tf.constant(42.0)}