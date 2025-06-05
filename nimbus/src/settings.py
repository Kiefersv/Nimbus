""" The functions in this file allow to adjust default settings of Nimbus"""

def set_solver_settings(
        self, initial_time_for_solver=None, end_time_for_solver=None,
        evaluation_steps_for_solver=20):
    if not isinstance(initial_time_for_solver, type(None)):
        self.tstart = initial_time_for_solver
    if not isinstance(end_time_for_solver, type(None)):
        self.tend = end_time_for_solver
    if not isinstance(evaluation_steps_for_solver, type(None)):
        self.tsteps = evaluation_steps_for_solver

def set_cloud_settings(
        self, minimum_cloud_particle_radius=None, molecular_cross_section=None,
):
    if not isinstance(minimum_cloud_particle_radius, type(None)):
        self.r_ccn = minimum_cloud_particle_radius
    if not isinstance(molecular_cross_section, type(None)):
        self.cs_mol = molecular_cross_section