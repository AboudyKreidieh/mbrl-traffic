"""Script containing the fully-connected model object."""
from mbrl_traffic.models.base import Model


class FeedForwardModel(Model):
    """TODO

    Attributes
    ----------
    model_lr : int
        the model learning rate
    batch_size : int
        the size of the batch for learning the model
    layer_norm : bool
        enable layer normalisation
    layers : list of int
        the size of the neural network for the model
    act_fun : tf.nn.*
        the activation function to use in the neural network
    stochastic : bool
        whether the output from the model is stochastic or deterministic
    num_ensembles : int
        number of ensemble models
    num_particles : int
         number of particles used to generate the forward estimate of the model
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 replay_buffer,
                 verbose,
                 model_lr,
                 batch_size,
                 layer_norm,
                 layers,
                 act_fun,
                 stochastic,
                 num_ensembles,
                 num_particles):
        """Instantiate the fully-connected model object.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        replay_buffer : mbrl_traffic.utils.replay_buffer.ReplayBuffer
            the replay buffer object used by the algorithm to store environment
            data
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        model_lr : int
            the model learning rate
        batch_size : int
            the size of the batch for learning the model
        layer_norm : bool
            enable layer normalisation
        layers : list of int
            the size of the neural network for the model
        act_fun : tf.nn.*
            the activation function to use in the neural network
        stochastic : bool
            whether the output from the model is stochastic or deterministic
        num_ensembles : int
            number of ensemble models
        num_particles : int
             number of particles used to generate the forward estimate of the
             model.
        """
        super(FeedForwardModel, self).__init__(
            sess, ob_space, ac_space, replay_buffer, verbose)

        self.model_lr = model_lr
        self.batch_size = batch_size
        self.layer_norm = layer_norm
        self.layers = layers
        self.act_fun = act_fun
        self.stochastic = stochastic
        self.num_ensembles = num_ensembles
        self.num_particles = num_particles

    def initialize(self):
        """See parent class."""
        pass  # FIXME

    def get_next_obs(self, obs, action):
        """See parent class."""
        pass  # FIXME

    def update(self):
        """See parent class."""
        pass  # FIXME

    def get_td_map(self):
        """See parent class."""
        pass  # FIXME
