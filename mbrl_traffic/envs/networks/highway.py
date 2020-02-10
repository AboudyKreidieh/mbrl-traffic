from flow.networks.base import Network


class I210SubNetwork(Network):
    """TODO

    Attributes
    ----------
    TODO
        TODO
    """

    def __init__(self, name, vehicles, net_params):
        """TODO

        Parameters
        ----------
        name : TODO
            TODO
        vehicles : TODO
            TODO
        net_params : TODO
            TODO
        """
        super().__init__(name, vehicles, net_params)

    def specify_nodes(self, net_params):
        """TODO

        Parameters
        ----------
        net_params : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        pass

    def specify_edges(self, net_params):
        """TODO

        Parameters
        ----------
        net_params : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        pass

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """TODO

        Parameters
        ----------
        cls : TODO
            TODO
        net_params : TODO
            TODO
        initial_config : TODO
            TODO
        num_vehicles : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        """
        pass
