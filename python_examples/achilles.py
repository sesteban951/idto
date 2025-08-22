#!/usr/bin/env python

##
#
# A simple sanity check with Adrian's Achilles humanoid.
#
##

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    AddDefaultVisualization,
    Parser,
    Box,
    RigidTransform,
    CoulombFriction,
)
import time
import numpy as np

from pyidto import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSolution,
    TrajectoryOptimizerStats,
    SolverParameters,
    ProblemDefinition,
    FindIdtoResource,
)

def visualize_trajectory(q, time_step, model_file, meshcat=None):
    """
    Display the given trajectory (list of configurations) on meshcat
    """
    # Create a simple Drake diagram with a plant model
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    Parser(plant).AddModels(model_file)
    plant.Finalize()

    # Connect to the meshcat visualizer
    AddDefaultVisualization(builder, meshcat)

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.get_actuation_input_port().FixValue(plant_context,
                                              np.zeros(plant.num_actuators()))

    # Step through q, setting the plant positions at each step
    meshcat.StartRecording()
    for k in range(len(q)):
        diagram_context.SetTime(k * time_step)
        plant.SetPositions(plant_context, q[k])
        diagram.ForcedPublish(diagram_context)
        time.sleep(time_step)
    meshcat.StopRecording()
    meshcat.PublishRecording()

if __name__=="__main__":
    # model_file = FindIdtoResource("../..idto/models/achilles/achilles.urdf")
    # model_file ="models/achilles/achilles.urdf"
    model_file ="models/achilles/achilles_lowered_com_walking_hardware_obj.urdf"

    # Create the system diagram that the optimizer uses
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.05)
    Parser(plant).AddModels(model_file)
    plant.RegisterCollisionGeometry(
        plant.world_body(), 
        RigidTransform(p=[0, 0, -25]), 
        Box(50, 50, 50), "ground", 
        CoulombFriction(0.5, 0.5))
    plant.Finalize()  # TODO: add a ground
    diagram = builder.Build()

    nq = plant.num_positions()
    nv = plant.num_velocities()

    q_stand = np.array([
        # 1.0000, 0.0000, 0.0000, 0.0000,           # base orientation
        # 0.0000, 0.0000, 0.9300,                   # base position
        0.0000, 0.0000, 0.0000, 1.0000,           # base orientation
        0.0000, 0.0000, 0.6500,                   # base position
        0.0000, 0.0209,-0.5515, 1.0239,-0.4725,   # left leg
        0.0000, 0.0000, 0.0000, 0.0000,           # left arm
        0.0000,-0.0209,-0.3200, 0.9751,-0.6552,   # right leg
        0.0000, 0.0000, 0.0000, 0.0000,           # right arm
    ])

    # Specify a cost function and target trajectory
    problem = ProblemDefinition()
    problem.num_steps = 40
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    problem.Qq = np.diag([
        10.0, 10.0, 10.0, 10.0,   # base orientation
        10.0, 10.0, 10.0,         # base position
        0.1, 0.1, 0.1, 0.1, 0.01,  # left leg
        0.01, 0.01, 0.01, 0.01,       # left arm
        0.1, 0.1, 0.1, 0.1, 0.01,  # right leg
        0.01, 0.01, 0.01, 0.01        # right arm
    ])
    problem.Qv = 0.01 * np.eye(nv)
    problem.R = 0.01 * np.diag([
        100.0, 100.0, 100.0, 100.0,    # base orientation
        100.0, 100.0, 100.0,           # base position
        0.01, 0.01, 0.01, 0.01, 0.01,  # left leg
        0.01, 0.01, 0.01, 0.01,        # left arm
        0.01, 0.01, 0.01, 0.01, 0.01,  # right leg
        0.01, 0.01, 0.01, 0.01,        # right arm
    ])
    problem.Qf_q = 10.0 * np.copy(problem.Qq)
    problem.Qf_v = 1.0 * np.copy(problem.Qv)

    base_vel = np.array([-0.5, 0.0])  # vx, vy
    q_sel = np.zeros(nq)
    q_sel[4:6] = base_vel
    v_nom = np.zeros(nv)
    v_nom[3:5] = base_vel
    dt = plant.time_step()
    problem.q_nom = [np.copy(q_stand) + q_sel * dt * i for i in range(problem.num_steps + 1)]
    problem.v_nom = [np.copy(v_nom) for i in range(problem.num_steps + 1)]

    # Set the solver parameters
    params = SolverParameters()
    params.max_iterations = 200
    params.scaling = True
    params.equality_constraints = False
    params.Delta0 = 1e1
    params.Delta_max = 1e5
    params.num_threads = 4
    params.contact_stiffness = 10_000
    params.dissipation_velocity = 0.1
    params.smoothing_factor = 0.01
    params.friction_coefficient = 0.5
    params.stiction_velocity = 0.2
    params.verbose = True

    # Specify an initial guess
    q_guess = [np.copy(q_stand) for i in range(problem.num_steps + 1)]

    # Solve the optimization problem
    optimizer = TrajectoryOptimizer(diagram, plant, problem, params)
    solution = TrajectoryOptimizerSolution()
    stats = TrajectoryOptimizerStats()
    optimizer.Solve(q_guess, solution, stats)
    solve_time = np.sum(stats.iteration_times)
    print("Solve time: ", solve_time)

    # Visualize the solution
    meshcat = StartMeshcat()
    visualize_trajectory(solution.q, 0.05, model_file, meshcat)
    time.sleep(10)  # Wait for the visualizer to catch up
