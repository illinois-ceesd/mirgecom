class InterfaceFluidRadiationBoundary(PrescribedFluidBoundary):
    """Interface boundary condition for the fluid side."""

    # FIXME: Incomplete docs
    def __init__(
            self, kappa_plus, t_plus, epsilon_plus, grad_t_plus=None,
            heat_flux_penalty_amount=None, lengthscales=None):
        """Initialize InterfaceFluidBoundary."""
        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=partial(
                _inviscid_flux_for_prescribed_state_mengaldo,
                state_plus_func=self.state_plus),
            viscous_flux_func=partial(
                _interface_viscous_flux_with_radiation,
                penalty_amount=heat_flux_penalty_amount,
                lengthscales=lengthscales,
                epsilon_plus=epsilon_plus,
                state_bc_func=self.state_bc,
                temperature_plus_func=self.temperature_plus,
                grad_cv_bc_func=self.grad_cv_bc,
                grad_temperature_bc_func=self.grad_temperature_bc),
            boundary_temperature_func=self.temperature_plus,
            boundary_gradient_cv_func=self.grad_cv_bc,
            boundary_gradient_temperature_func=self.grad_temperature_bc)

        self._thermally_coupled = _ThermallyCoupledUpwindBoundaryComponent(
            kappa_plus=kappa_plus,
            t_plus=t_plus,
            grad_t_plus=grad_t_plus)
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)

        mom_plus = self._no_slip.momentum_plus(cv_minus.momentum, normal)

        # Don't modify the energy, even though t_plus != t_minus; energy will
        # be advected in/out of the wall, which doesn't make sense
        cv_plus = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=cv_minus.energy,
            momentum=mom_plus,
            species_mass=cv_minus.species_mass)

        kappa_plus = self._thermally_coupled.kappa_plus(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_plus, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_plus)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        dd_bdry = as_dofdesc(dd_bdry)
        normal = actx.thaw(dcoll.normal(dd_bdry))

        cv_minus = state_minus.cv

        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)

        mom_bc = self._no_slip.momentum_bc(cv_minus.momentum, normal)

        t_bc = self._thermally_coupled.temperature_bc(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=cv_minus.species_mass_fractions)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = cv_minus.mass * internal_energy_bc

        cv_bc = make_conserved(
            state_minus.dim,
            mass=cv_minus.mass,
            energy=total_energy_bc,
            momentum=mom_bc,
            species_mass=cv_minus.species_mass)

        kappa_bc = self._thermally_coupled.kappa_bc(dcoll, dd_bdry, kappa_minus)

        return _replace_kappa(
            make_fluid_state(
                cv=cv_bc, gas_model=gas_model,
                temperature_seed=state_minus.temperature),
            kappa_bc)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, state_bc, grad_cv_minus,
            **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_cv_minus.momentum,
            species_mass=grad_species_mass_bc)

    def temperature_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior T on the boundary."""
        kappa_minus = (
            # Make sure it has an array context
            state_minus.tv.thermal_conductivity + 0*state_minus.mass_density)
        return self._thermally_coupled.temperature_plus(
            dcoll, dd_bdry, kappa_minus, state_minus.temperature)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, **kwargs):
        """Get grad(T) on the boundary."""
        return self._thermally_coupled.grad_temperature_bc(
            dcoll, dd_bdry, grad_t_minus)
