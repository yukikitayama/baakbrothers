import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def dqn_cartpole_model(num_states, num_actions):
    inputs = Input(shape=(num_states,))
    out = Dense(128, activation='relu')(inputs)
    out = Dense(128, activation='relu')(out)
    outputs = Dense(num_actions, activation='linear')(out)
    model = Model(inputs, outputs)
    return model


def get_action(state, num_actions, model, epsilon):
    # Exploration by epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Exploitation by greedy
    else:
        return np.argmax(model.predict(np.atleast_2d(state))[0])


def update_target(non_target_model, target_model):
    depth = len(target_model.trainable_variables)
    for i in range(depth):
        weights_non_target = non_target_model.trainable_variables[i]
        target_model.trainable_variables[i].assign(weights_non_target.numpy())
    return target_model


def update_model(model, target_model, memory, min_experience, batch_size,
                 gamma, num_actions, optimizer):
    if memory.size() < min_experience:
        return 0

    # Get mini batch
    index = np.random.randint(low=0, high=len(memory.buffer['state']),
                              size=batch_size)
    states = np.asarray([memory.buffer['state'][i] for i in index])
    actions = np.asarray([memory.buffer['action'][i] for i in index])
    rewards = np.asarray([memory.buffer['reward'][i] for i in index])
    next_states = np.asarray([memory.buffer['next_state'][i] for i in index])
    dones = np.asarray([memory.buffer['done'][i] for i in index])

    next_action_values = np.max(target_model.predict(next_states), axis=1)
    # np.where allows us to have if the first argument is true, choose the
    # second argument, otherwise choose the third argument
    # done = True means it's a terminal state, so we only have a reward, and
    # no discounted action values from the next state.
    target_values = np.where(dones,
                             rewards,
                             rewards + gamma * next_action_values)

    # Update neural network weights
    with tf.GradientTape() as tape:
        action_values = tf.math.reduce_sum(
            model(np.atleast_2d(states.astype('float32'))) * tf.one_hot(actions, num_actions),
            axis=1
        )
        # Q network is trained byb minimising loss function
        loss = tf.math.reduce_mean(tf.square(target_values - action_values))
    # Gradient descent by differentiating loss function w.r.t. weights
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, variables))
    return loss