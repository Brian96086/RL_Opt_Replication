from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.models.modelv2 import ModelV2

import numpy as np
tf = try_import_tf()[0]

torch, nn = try_import_torch()

class CustomDQNModel(DQNTorchModel):
    def __init__(self, observation_space, action_space, num_outputs, model_config,
                 name="my_model",**kw):
        #import ipdb;ipdb.set_trace()
        # print("MyKerasQModel keys = {0}".format(model_config.keys()))
        # model_config["parameter_noise"] = kw["parameter_noise"]
        kw.pop("framework", None)
        super(CustomDQNModel, self).__init__(
            observation_space, action_space, num_outputs, model_config, name, **kw)
        # self.inputs = tf.keras.layers.Input(
        #     shape=observation_space.shape, name="observations")
        # layer_1 = tf.keras.layers.Dense(
        #     12,
        #     name="my_layer1",
        #     activation=tf.nn.tanh,
        #     kernel_initializer=tf.keras.initializers.he_normal())(self.inputs)
        # layer_2 = tf.keras.layers.Dense(
        #     8,
        #     name="my_layer2",
        #     activation=tf.nn.tanh,
        #     kernel_initializer=tf.keras.initializers.he_normal())(layer_1)
        # layer_out = tf.keras.layers.Dense(
        #     num_outputs,
        #     name="my_out",
        #     activation=tf.nn.softmax,
        #     kernel_initializer=tf.keras.initializers.he_normal())(layer_2)

        # base model : 8(obs.shape) -> 12 -> 2(Q for two actions of lockdown)
        #print(f'obs_space shape = {observation_space.shape}')
        self.base_model = nn.Sequential(
            nn.Linear(torch.tensor(observation_space.shape), 12),
            nn.Tanh(),
            nn.Linear(12, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )
        # print("num_outputs = ", num_outputs)
        # print("layer_1 = ", layer_1.shape)
        # print("layer_2 = ", layer_2.shape)
        # print("layer_out = ", layer_out.shape)
        # self.register_variables(self.base_model.variables)
    
    def forward(self, input_dict, state, seq_lens):
        #print(f'original obs(forward) = {type(input_dict["obs"])}, {input_dict["obs"].values}')
        obs = np.array(input_dict["obs"])
        print(f'input obs(forward) = {obs}, {type(obs)}')
        model_out = self.base_model(obs)
        return model_out, state

    def metrics(self):
        #return {"foo": torch.constant(42.0)}
        return {"foo": torch.tensor(42.0)}