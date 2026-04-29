class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend

    def __del__(self):
        self.backend.DeleteBootstrappers()

    def generate_bootstrapper(self, slots):
        # We will wait to instantiate any bootstrapper until our bootstrap
        # placement algorithm determines they're necessary.
        logp = self.scheme.params.get_boot_logp()
        return self.backend.NewBootstrapper(logp, slots)
    
    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)

    def bootstrap_many(self, ctxts, slots):
        bootstrap_many = getattr(self.backend, "BootstrapMany", None)
        if bootstrap_many is None:
            return [self.bootstrap(ctxt, slots) for ctxt in ctxts]
        return bootstrap_many(list(ctxts), slots)
