""":mod:`mirgecom.flux` provides inter-facial flux routines.

Numerical Flux Routines
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradient_flux_central
.. autofunction:: divergence_flux_central
.. autofunction:: flux_lfr
.. autofunction:: divergence_flux_lfr
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np  # noqa


def gradient_flux_central(u_tpair, normal):
    r"""Compute a central flux for the gradient operator.

    The central gradient flux, $\mathbf{h}$, of a scalar quantity $u$ is calculated
    as:

    .. math::

        \mathbf{h}({u}^-, {u}^+; \mathbf{n}) = \frac{1}{2}
        \left({u}^{+}+{u}^{-}\right)\mathbf{\hat{n}}

    where ${u}^-, {u}^+$, are the scalar function values on the interior
    and exterior of the face on which the central flux is to be calculated, and
    $\mathbf{\hat{n}}$ is the *normal* vector.

    *u_tpair* is the :class:`~grudge.trace_pair.TracePair` representing the scalar
    quantities ${u}^-, {u}^+$. *u_tpair* may also represent a vector-quantity
    :class:`~grudge.trace_pair.TracePair`, and in this case the central scalar flux
    is computed on each component of the vector quantity as an independent scalar.

    Parameters
    ----------
    u_tpair: :class:`~grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with the flux for each
        scalar component.
    """
    from arraycontext import outer
    return outer(u_tpair.avg, normal)


def divergence_flux_central(trace_pair, normal):
    r"""Compute a central flux for the divergence operator.

    The central divergence flux, $h$, is calculated as:

    .. math::

        h(\mathbf{v}^-, \mathbf{v}^+; \mathbf{n}) = \frac{1}{2}
        \left(\mathbf{v}^{+}+\mathbf{v}^{-}\right) \cdot \hat{n}

    where $\mathbf{v}^-, \mathbf{v}^+$, are the vectors on the interior and exterior
    of the face across which the central flux is to be calculated, and $\hat{n}$ is
    the unit normal to the face.

    Parameters
    ----------
    trace_pair: :class:`~grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed
    normal: numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray` with the flux for each
        scalar component.
    """
    return trace_pair.avg@normal


def divergence_flux_lfr(cv_tpair, f_tpair, normal, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+)) \cdot
        \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{n}$ is the face normal, and $\lambda$ is the
    user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux Jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`~meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov flux.
    """
    from arraycontext import outer
    #return flux_lfr(cv_tpair, f_tpair, normal, lam)@normal
    return flux_lfr(cv_tpair, f_tpair, normal, lam)


def flux_lfr(cv_tpair, f_tpair, normal, lam):
    r"""Compute Lax-Friedrichs/Rusanov flux after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+))
        + \frac{\lambda}{2}(q^{-} - q^{+})\hat{\mathbf{n}},

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{\mathbf{n}}$ is the face normal, and $\lambda$
    is the user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    lam: :class:`~meshmode.dof_array.DOFArray`

        lambda parameter for Lax-Friedrichs/Rusanov flux

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        Lax-Friedrichs/Rusanov flux.
    """
    from arraycontext import outer
    #return f_tpair.avg - lam*outer(cv_tpair.diff, normal)/2.
    return lfr(f_tpair.ext@normal, f_tpair.int@normal, cv_tpair.ext, cv_tpair.int, lam)

def lfr(f_plus, f_minus, cv_plus, cv_minus, lam):
    from arraycontext import outer
    #return (f_plus + f_minus - lam*outer(cv_plus - cv_minus, normal))/2.
    return (f_plus + f_minus - lam*(cv_plus - cv_minus))/2.


def divergence_flux_hll(cv_tpair, normal, eos):
    r"""Compute Harten, Lax, van Leer Contact (HLLC) flux
        after [Hesthaven_2008]_, Section 6.6.


    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLLC flux.
    """
    #print(f"{normal=}")
    return flux_hll(cv_tpair, normal, eos)@normal
    #return flux_hll(cv_tpair, f_tpair, normal, eos)


def flux_hll(cv_tpair, normal, eos):
    r"""Compute Harten, Lax, van Leer Contact (HLLC) flux
        after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:
    Update this math brother!

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+))
        + \frac{\lambda}{2}(q^{-} - q^{+})\hat{\mathbf{n}},

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{\mathbf{n}}$ is the face normal, and $\lambda$
    is the user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLLC flux.
    """
    from arraycontext import outer
    actx = cv_tpair.int.array_context
    dim = cv_tpair.int.dim
    zeros = 0.*cv_tpair.int.mass
    ones = zeros + 1.
    #lnorm = np.zeros(dim, dtype=object)
    lnorm = 0.*cv_tpair.int.velocity
    lnorm[0] = ones

    # first rotate the 2D/3D problem into a 1D problem in the normal direction with 
    # respect to the interface

    from mirgecom.fluid import make_conserved
    cv_int = make_conserved(dim=dim, mass=cv_tpair.int.mass,
                                momentum=np.dot(cv_tpair.int.momentum, normal)*lnorm,
                                energy=cv_tpair.int.energy,
                                species_mass=cv_tpair.int.species_mass
                                )
    cv_ext = make_conserved(dim=dim, mass=cv_tpair.ext.mass,
                                momentum=np.dot(cv_tpair.ext.momentum, normal)*lnorm,
                                energy=cv_tpair.ext.energy,
                                species_mass=cv_tpair.ext.species_mass
                                )

    # note for me, treat the interior state as left and the exterior state as right
    # pressure estimate
    p_int = eos.pressure(cv_int)
    p_ext = eos.pressure(cv_ext)
    u_int = np.dot((cv_int).velocity, lnorm)
    #print(f"{u_int=}")
    #print(f"{cv_int.momentum=}")
    u_ext = np.dot((cv_ext).velocity, lnorm)
    rho_int = (cv_int).mass
    rho_ext = (cv_ext).mass
    #print(f"{p_int=}")
    #print(f"{rho_int=}")
    c_int = eos.sound_speed(cv_int)
    c_ext = eos.sound_speed(cv_ext)


    #umag_int = actx.np.sqrt(np.dot(u_int, u_int))/rho_int
    #umag_ext = actx.np.sqrt(np.dot(u_ext, u_ext))/rho_ext
    #umag_int = actx.np.sqrt(np.dot(u_int, u_int))
    #umag_ext = actx.np.sqrt(np.dot(u_ext, u_ext))
    #print(f"{u_int=}")
    #print(f"u_int*normal {u_int@normal}")
    #umag_int = u_int[0]
    #umag_ext = u_ext[0]
    #umag_int = np.dot(u_int, normal)
    #umag_ext = np.dot(u_ext, normal)
    #print(f"{umag_int=}")

    p_star = (0.5*(p_int + p_ext) + (1./8.)*(u_int - u_ext)*
             (rho_int + rho_ext)*(c_int + c_ext))
    #p_star = 0.5*(p_int + p_ext)
    #print(f"p_star {p_star}")

    # left and right wave speeds
    q_int = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_int - 1)
    q_ext = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_ext - 1)

    pres_check_int = actx.np.greater(p_star, p_int)
    pres_check_ext = actx.np.greater(p_star, p_ext)

    q_int = actx.np.where(pres_check_int, q_int, ones)
    q_ext = actx.np.where(pres_check_ext, q_ext, ones)

    q_int = actx.np.sqrt(q_int)
    q_ext = actx.np.sqrt(q_ext)

    # left, right, and intermediate wave speed estimates
    # can alternatively use the roe estimated states to find the wave speeds
    #print(f"c_int {c_int}")
    #print(f"q_int {q_int}")
    s_int = u_int - c_int*q_int
    s_ext = u_ext + c_ext*q_ext

    #print(f"s_int {s_int}")
    #print(f"s_ext {s_ext}")

    # HLL fluxes
    #flux_int = f_tpair.int
    #flux_ext = f_tpair.ext

    #flux_int = make_conserved(dim=dim, mass=f_tpair.int.mass,
                              #momentum=np.dot(f_tpair.int.momentum, normal),
                              #energy=f_tpair.int.energy,
                              #species_mass=f_tpair.int.species_mass
                             #)
    #flux_ext = make_conserved(dim=dim, mass=f_tpair.ext.mass,
                              #momentum=np.dot(f_tpair.ext.momentum, normal),
                              #energy=f_tpair.ext.energy,
                              #species_mass=f_tpair.ext.species_mass
                             #)
    #flux_int = outer(f_tpair.int, normal)
    #flux_ext = outer(f_tpair.ext, normal)

    from mirgecom.inviscid import inviscid_flux
    flux_int = inviscid_flux(p_int, cv_int)
    flux_ext = inviscid_flux(p_ext, cv_ext)

    flux_star = (s_ext*flux_int-s_int*flux_ext+s_ext*s_int*(cv_ext-cv_int))/(s_ext-s_int)

    # choose the correct flux contribution based on the wave speeds
    flux_check_int = actx.np.greater_equal(s_int, zeros)*(0*flux_int + 1.0)
    flux_check_ext = actx.np.less_equal(s_ext, zeros)*(0*flux_int + 1.0)

    #print(f"flux_check_int.mass {flux_check_int.mass}")
    #print(f"flux_check_ext.mass {flux_check_ext.mass}")

    flux = flux_star
    flux = actx.np.where(flux_check_int, flux_int, flux)
    flux = actx.np.where(flux_check_ext, flux_ext, flux)

    #print(f"flux_int.mass {flux_int.mass}")
    #print(f"flux_int.momentum {flux_int.momentum}")
    #print(f"flux_int.energy {flux_int.energy}")

    #print(f"flux_ext.mass {flux_ext.mass}")
    #print(f"flux_ext.momentum {flux_ext.momentum}")
    #print(f"flux_ext.energy {flux_ext.energy}")

    #print(f"flux.mass {flux.mass}")
    #print(f"flux.momentum {flux.momentum}")
    #print(f"flux.energy {flux.energy}")

    flux_rotated = make_conserved(dim=dim, mass=flux.mass*normal,
                                  #momentum=np.dot(flux.momentum, lnorm)*normal,
                                  #momentum=flux.momentum@normal,
                                  momentum=flux.momentum,
                                  energy=flux.energy*normal,
                                  species_mass=flux.species_mass*normal
                                 )
    #flux_rotated = outer(flux, normal)

    #print(f"flux.mass {flux.mass}")
    #print(f"flux_rotated.mass {flux_rotated.mass}")
    #print(f"flux.momentum {flux.momentum}")
    #print(f"flux_rotated.momentum {flux_rotated.momentum}")
    #print(f"flux.energy {flux.energy}")
    #print(f"flux_rotated.energy {flux_rotated.energy}")

    #return outer(flux, normal)
    return flux_rotated
    #return flux

def divergence_flux_hllc(cv_tpair, f_tpair, normal, eos):
    r"""Compute Harten, Lax, van Leer Contact (HLLC) flux
        after [Hesthaven_2008]_, Section 6.6.


    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLLC flux.
    """
    return flux_hllc(cv_tpair, normal, eos)@normal


def flux_hllc_old(cv_tpair, f_tpair, normal, eos):
    r"""Compute Harten, Lax, van Leer Contact (HLLC) flux
        after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:
    Update this math brother!

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+))
        + \frac{\lambda}{2}(q^{-} - q^{+})\hat{\mathbf{n}},

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{\mathbf{n}}$ is the face normal, and $\lambda$
    is the user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLLC flux.
    """
    from arraycontext import outer
    actx = cv_tpair.int.array_context
    dim = cv_tpair.int.dim

    # first rotate the 2D/3D problem in a 1D problem in the normal direction with 
    # respect to the interface

    # note for me, treat the interior state as left and the exterior state as right
    # pressure estimate
    p_int = eos.pressure(cv_tpair.int)
    p_ext = eos.pressure(cv_tpair.ext)
    u_int = (cv_tpair.int).velocity
    u_ext = (cv_tpair.ext).velocity
    rho_int = (cv_tpair.int).mass
    rho_ext = (cv_tpair.ext).mass
    print(f"{p_int=}")
    print(f"{rho_int=}")
    c_int = eos.sound_speed(cv_tpair.int)
    c_ext = eos.sound_speed(cv_tpair.ext)

    ones = (1.0 + p_ext) - p_ext
    zeros = 0.*p_ext

    umag_int = actx.np.sqrt(np.dot(u_int, u_int))/rho_int
    umag_ext = actx.np.sqrt(np.dot(u_ext, u_ext))/rho_ext

    p_star = (0.5*(p_int + p_ext) + 8*(umag_int - umag_ext)*
             (rho_int + rho_ext)*(c_int + c_ext))
    #p_star = 0.5*(p_int + p_ext)
    #print(f"p_star {p_star}")

    # left and right wave speeds
    q_int = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_int - 1)
    q_ext = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_ext - 1)

    pres_check_int = actx.np.greater(p_star, p_int)
    pres_check_ext = actx.np.greater(p_star, p_ext)

    q_int = actx.np.where(pres_check_int, q_int, ones)
    q_ext = actx.np.where(pres_check_ext, q_ext, ones)

    q_int = actx.np.sqrt(q_int)
    q_ext = actx.np.sqrt(q_ext)

    # left, right, and intermediate wave speed estimates
    # can alternatively use the roe estimated states to find the wave speeds
    print(f"c_int {c_int}")
    print(f"q_int {q_int}")
    s_int = umag_int - c_int*q_int
    s_ext = umag_ext + c_ext*q_ext
    s_star = ((p_ext - p_int + rho_int*umag_int*(s_int - umag_int) - rho_ext*umag_ext*(s_ext - umag_ext))/
             (rho_int*(s_int - umag_int) - rho_ext*(s_ext - umag_ext)))

    print(f"s_int {s_int}")
    print(f"s_ext {s_ext}")
    print(f"s_star {s_star}")

    # HLLC fluxes
    flux_int = f_tpair.int
    flux_ext = f_tpair.ext

    # for now, just 1D to see if it works
    #flux_star_int = f_tpair.int - s_int*cv_tpair.int
    flux_star_int_mass = s_int*rho_int*(s_int - u_int)/(s_int - s_star)
    flux_star_int_velocity = s_int*rho_int*(s_int - u_int)/(s_int - s_star)*s_star
    flux_star_int_energy = (s_int*rho_int*(s_int - u_int)/(s_int - s_star)*
              (cv_tpair.int.energy/rho_int + (s_star - u_int)*(s_star + p_int/(rho_int*(s_int - u_int)))))
    flux_star_int_species = s_int*(s_int - u_int)/(s_int - s_star)*cv_tpair.int.species_mass.reshape(-1,1)

    from mirgecom.fluid import make_conserved
    flux_star_int = make_conserved(dim, mass=flux_star_int_mass, momentum=flux_star_int_velocity,
                                   energy=flux_star_int_energy, species_mass=flux_star_int_species)
    flux_star_int += f_tpair.int - s_int*cv_tpair.int

    #flux_star_ext = f_tpair.ext - s_ext*cv_tpair.ext
    flux_star_ext_mass = s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)
    flux_star_ext_velocity = s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)*s_star
    flux_star_ext_energy = (s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)*
              (cv_tpair.ext.energy/rho_ext + (s_star - u_ext)*(s_star + p_ext/(rho_ext*(s_ext - u_ext)))))
    flux_star_ext_species = s_ext*(s_ext - u_ext)/(s_ext - s_star)*cv_tpair.ext.species_mass.reshape(-1,1)

    flux_star_ext = make_conserved(dim, mass=flux_star_ext_mass, momentum=flux_star_ext_velocity,
                                   energy=flux_star_ext_energy, species_mass=flux_star_ext_species)
    flux_star_ext += f_tpair.ext - s_ext*cv_tpair.ext

    # choose the correct flux contribution based on the wave speeds

    flux_check_int = actx.np.greater_equal(s_int, zeros)*(0*flux_int + 1.0)
    flux_check_ext = actx.np.less_equal(s_ext, zeros)*(0*flux_int + 1.0)
    flux_check_star_int = actx.np.less_equal(s_int, zeros)*actx.np.greater(s_star, zeros)*(0*flux_int + 1.0)
    flux_check_star_ext = actx.np.less_equal(s_star, zeros)*actx.np.greater_equal(s_ext, zeros)*(0*flux_int + 1.0)


    print(f"flux_check_int.mass {flux_check_int.mass}")
    print(f"flux_check_ext.mass {flux_check_ext.mass}")
    print(f"flux_check_star_int.mass {flux_check_star_int.mass}")
    print(f"flux_check_star_ext.mass {flux_check_star_ext.mass}")

    flux = f_tpair.ext*0
    flux = actx.np.where(flux_check_int, flux_int, flux)
    flux = actx.np.where(flux_check_ext, flux_ext, flux)
    flux = actx.np.where(flux_check_star_int, flux_star_int, flux)
    flux = actx.np.where(flux_check_star_ext, flux_star_ext, flux)

    #
    # test code, try multiplying the bools together
    #
    #flux_check_int = actx.np.greater_equal(s_int, zeros)
    #flux_check_ext = actx.np.less_equal(s_ext, zeros)
    #flux_check_star_int = actx.np.less_equal(s_int, zeros)*actx.np.greater_equal(s_star, zeros)
    #flux_check_star_ext = actx.np.less_equal(s_star, zeros)*actx.np.greater_equal(s_ext, zeros)

    #print(f"flux_check_int {flux_check_int}")
    #print(f"flux_check_ext {flux_check_ext}")
    #print(f"flux_check_star_int {flux_check_star_int}")
    #print(f"flux_check_star_ext {flux_check_star_ext}")

    #flux = ((flux_int*flux_check_int
           #+ flux_ext*flux_check_ext
           #+ flux_star_int*flux_check_star_int
           #+ flux_star_ext*flux_check_star_ext)/
           #(flux_check_int + flux_check_ext +
            #flux_check_star_int + flux_check_star_ext))

    #print(f"flux_int.mass {flux_int.mass}")
    #print(f"flux_int.momentum {flux_int.momentum}")
    #print(f"flux_int.energy {flux_int.energy}")

    #print(f"flux_ext.mass {flux_ext.mass}")
    #print(f"flux_ext.momentum {flux_ext.momentum}")
    #print(f"flux_ext.energy {flux_ext.energy}")

    #print(f"flux_star_int.mass {flux_star_int.mass}")
    #print(f"flux_star_int.momentum {flux_star_int.momentum}")
    #print(f"flux_star_int.energy {flux_star_int.energy}")

    #print(f"flux_star_ext.mass {flux_star_ext.mass}")
    #print(f"flux_star_ext.momentum {flux_star_ext.momentum}")
    #print(f"flux_star_ext.energy {flux_star_ext.energy}")

    #print(f"flux.mass {flux.mass}")
    #print(f"flux.momentum {flux.momentum}")
    #print(f"flux.energy {flux.energy}")

    return flux

def flux_hllc(cv_tpair, normal, eos):
    r"""Compute Harten, Lax, van Leer Contact (HLLC) flux
        after [Hesthaven_2008]_, Section 6.6.

    The Lax-Friedrichs/Rusanov flux is calculated as:
    Update this math brother!

    .. math::

        f_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-) + \mathbf{F}(q^+))
        + \frac{\lambda}{2}(q^{-} - q^{+})\hat{\mathbf{n}},

    where $q^-, q^+$ are the scalar solution components on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the vector flux function, $\hat{\mathbf{n}}$ is the face normal, and $\lambda$
    is the user-supplied jump term coefficient.

    The $\lambda$ parameter is system-specific. Specifically, for the Rusanov flux
    it is the max eigenvalue of the flux jacobian:

    .. math::
        \lambda = \text{max}\left(|\mathbb{J}_{F}(q^-)|,|\mathbb{J}_{F}(q^+)|\right)

    Here, $\lambda$ is a function parameter, leaving the responsibility for the
    calculation of the eigenvalues of the system-dependent flux Jacobian to the
    caller.

    Parameters
    ----------
    cv_tpair: :class:`~grudge.trace_pair.TracePair`

        Solution trace pair for faces for which numerical flux is to be calculated

    f_tpair: :class:`~grudge.trace_pair.TracePair`

        Physical flux trace pair on faces on which numerical flux is to be calculated

    normal: numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with outward-pointing
        normals

    Returns
    -------
    numpy.ndarray

        object array of :class:`~meshmode.dof_array.DOFArray` with the
        HLLC flux.
    """
    from arraycontext import outer
    actx = cv_tpair.int.array_context
    dim = cv_tpair.int.dim
    zeros = 0.*cv_tpair.int.mass
    ones = zeros + 1.
    #lnorm = np.zeros(dim, dtype=object)
    lnorm = 0.*cv_tpair.int.velocity
    lnorm[0] = ones

    # first rotate the 2D/3D problem into a 1D problem in the normal direction with 
    # respect to the interface

    from mirgecom.fluid import make_conserved
    cv_int = make_conserved(dim=dim, mass=cv_tpair.int.mass,
                                momentum=np.dot(cv_tpair.int.momentum, normal)*lnorm,
                                energy=cv_tpair.int.energy,
                                species_mass=cv_tpair.int.species_mass
                                )
    cv_ext = make_conserved(dim=dim, mass=cv_tpair.ext.mass,
                                momentum=np.dot(cv_tpair.ext.momentum, normal)*lnorm,
                                energy=cv_tpair.ext.energy,
                                species_mass=cv_tpair.ext.species_mass
                                )

    # note for me, treat the interior state as left and the exterior state as right
    # pressure estimate
    p_int = eos.pressure(cv_int)
    p_ext = eos.pressure(cv_ext)
    u_int = np.dot((cv_int).velocity, lnorm)
    u_ext = np.dot((cv_ext).velocity, lnorm)
    rho_int = (cv_int).mass
    rho_ext = (cv_ext).mass
    c_int = eos.sound_speed(cv_int)
    c_ext = eos.sound_speed(cv_ext)

    p_star = (0.5*(p_int + p_ext) + (1./8.)*(u_int - u_ext)*
             (rho_int + rho_ext)*(c_int + c_ext))

    # left and right wave speeds
    q_int = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_int - 1)
    q_ext = 1 + (eos.gamma()+1)/(2*eos.gamma())*(p_star/p_ext - 1)

    pres_check_int = actx.np.greater(p_star, p_int)
    pres_check_ext = actx.np.greater(p_star, p_ext)

    q_int = actx.np.where(pres_check_int, q_int, ones)
    q_ext = actx.np.where(pres_check_ext, q_ext, ones)

    q_int = actx.np.sqrt(q_int)
    q_ext = actx.np.sqrt(q_ext)

    # left, right, and intermediate wave speed estimates
    s_int = u_int - c_int*q_int
    s_ext = u_ext + c_ext*q_ext

    # can alternatively use the roe estimated states to find the wave speeds
    #h_int = (cv_int.energy+p_int)/rho_int
    #h_ext = (cv_ext.energy+p_ext)/rho_ext
    #u_roe = ((actx.np.sqrt(rho_int)*u_int + actx.np.sqrt(rho_ext)*u_ext)/
             #(actx.np.sqrt(rho_int) + actx.np.sqrt(rho_ext)))
    #h_roe = ((actx.np.sqrt(rho_int)*h_int + actx.np.sqrt(rho_ext)*h_ext)/
             #(actx.np.sqrt(rho_int) + actx.np.sqrt(rho_ext)))
    #c_roe = actx.np.sqrt((eos.gamma()-1)*(h_rho - 0.5*np.dot(u_int, u_int)))
#
    #s_int_roe = u_roe - c_roe
    #s_ext_roe = u_roe + c_roe
                #
    #s_int = s_int_roe
    #s_ext = s_ext_roe

    # HLL fluxes
    from mirgecom.inviscid import inviscid_flux
    flux_int = inviscid_flux(p_int, cv_int)
    flux_ext = inviscid_flux(p_ext, cv_ext)

    # for now, just 1D to see if it works
    flux_star_int_mass = s_int*rho_int*(s_int - u_int)/(s_int - s_star)
    flux_star_int_velocity = s_int*rho_int*(s_int - u_int)/(s_int - s_star)*s_star
    flux_star_int_energy = (s_int*rho_int*(s_int - u_int)/(s_int - s_star)*
              (cv_int.energy/rho_int + (s_star - u_int)*(s_star + p_int/(rho_int*(s_int - u_int)))))
    flux_star_int_species = s_int*(s_int - u_int)/(s_int - s_star)*cv_int.species_mass.reshape(-1,1)

    from mirgecom.fluid import make_conserved
    flux_star_int = make_conserved(dim, mass=flux_star_int_mass, momentum=flux_star_int_velocity,
                                   energy=flux_star_int_energy, species_mass=flux_star_int_species)
    flux_star_int += f_tpair.int - s_int*cv_int

    flux_star_ext_mass = s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)
    flux_star_ext_velocity = s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)*s_star
    flux_star_ext_energy = (s_ext*rho_ext*(s_ext - u_ext)/(s_ext - s_star)*
              (cv_ext.energy/rho_ext + (s_star - u_ext)*(s_star + p_ext/(rho_ext*(s_ext - u_ext)))))
    flux_star_ext_species = s_ext*(s_ext - u_ext)/(s_ext - s_star)*cv_ext.species_mass.reshape(-1,1)

    flux_star_ext = make_conserved(dim, mass=flux_star_ext_mass, momentum=flux_star_ext_velocity,
                                   energy=flux_star_ext_energy, species_mass=flux_star_ext_species)
    flux_star_ext += f_tpair.ext - s_ext*cv_ext

    # choose the correct flux contribution based on the wave speeds

    flux_check_int = actx.np.greater_equal(s_int, zeros)*(0*flux_int + 1.0)
    flux_check_ext = actx.np.less_equal(s_ext, zeros)*(0*flux_int + 1.0)
    flux_check_star_int = actx.np.less_equal(s_int, zeros)*actx.np.greater(s_star, zeros)*(0*flux_int + 1.0)
    flux_check_star_ext = actx.np.less_equal(s_star, zeros)*actx.np.greater_equal(s_ext, zeros)*(0*flux_int + 1.0)

    #print(f"flux_check_int.mass {flux_check_int.mass}")
    #print(f"flux_check_ext.mass {flux_check_ext.mass}")
    #print(f"flux_check_star_int.mass {flux_check_star_int.mass}")
    #print(f"flux_check_star_ext.mass {flux_check_star_ext.mass}")

    flux = f_tpair.ext*0
    flux = actx.np.where(flux_check_int, flux_int, flux)
    flux = actx.np.where(flux_check_ext, flux_ext, flux)
    flux = actx.np.where(flux_check_star_int, flux_star_int, flux)
    flux = actx.np.where(flux_check_star_ext, flux_star_ext, flux)

    flux_rotated = make_conserved(dim=dim, mass=flux.mass*normal,
                                  #momentum=np.dot(flux.momentum, lnorm)*normal,
                                  #momentum=flux.momentum@normal,
                                  momentum=flux.momentum,
                                  energy=flux.energy*normal,
                                  species_mass=flux.species_mass*normal
                                 )

    return flux_rotated
