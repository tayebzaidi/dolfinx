from dolfin import cpp, function
import typing

class MultiPointConstraint(cpp.fem.MultiPointConstraint):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace],
            master_slave_map: typing.Dict[int, int]):

        """Representation of MultiPointConstraint which is imposed on
        a linear system.

        """

        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V

        super().__init__(_V, master_slave_map)
