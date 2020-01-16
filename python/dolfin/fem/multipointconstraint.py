from dolfin import cpp, function
import typing


class MultiPointConstraint(cpp.fem.MultiPointConstraint):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace],
            master_slave_map: typing.Dict[int, typing.Dict[int, float]]):

        """Representation of MultiPointConstraint which is imposed on
        a linear system.

        """
        slaves = []
        masters = []
        coefficients = []
        offsets = []
        compat_map = {}

        for slave in master_slave_map.keys():
            offsets.append(len(masters))
            slaves.append(slave)
            for master in master_slave_map[slave]:
                compat_map[slave] = master
                break
                masters.append(master)
                coefficients.append(master_slave_map[slave][master])
        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V
        #compat map tmp struct until everything is rewritten
        super().__init__(_V, compat_map)#slaves, masters, coefficients, offsets)
