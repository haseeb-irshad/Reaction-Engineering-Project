class Benchmark:
    """Base class for reaction process benchmarks.
    """

    def __init__(
        self,
        structure_params,
        physics_params,
        kinetics_params,
        transport_params,
        operation_params,
        operation_name2ind,
        measure_ind2name,
        var2unit
    ):
        self._structure_params = structure_params
        self._physics_params = physics_params
        self._kinetics_params = kinetics_params
        self._transport_params = transport_params
        self._operation_params = operation_params
        self._operation_name2ind = operation_name2ind
        self._measure_ind2name = measure_ind2name
        self._var2unit = var2unit

    def named_operation_params(self):
        return list(self.name_to_ind.keys())

    def structure_params(self):
        return self._structure_params

    def physics_params(self):
        return self._physics_params

    def kinetics_params(self):
        return self._kinetics_params

    def transport_params(self):
        return self._transport_params

    def operation_params(self):
        return self._operation_params

    def operation_name2ind(self):
        return self._operation_name2ind

    def measure_ind2name(self):
        return self._measure_ind2name

    def var2unit(self):
        return self._var2unit
    
    def params(self):
        params = {}
        params.update(self._structure_params)
        params.update(self._physics_params)
        params.update(self._kinetics_params)
        params.update(self._transport_params)
        params.update(self._operation_params)
        return params
