###########################################################################
import math
import random
import time
import wandb
import cycle 
from controller import Robot, GPS, Supervisor
###########################################################################
## Constants
###########################################################################
WANDB = False
WANDB = True
NUM_GENERATIONS = 200
POPULATION_SIZE = 100
MUTATION_RATE = 0.003
PARAMS = 10           # number of controlled motors
NUM_ACTIVATIONS = 20  # number of actions (gait cycles per individual)
TIME_STEP = 20        # default time step
HEIGHT_WEIGHT = 10    # weight for the height component of the fitness
"""JOINT_LIMITS = {      # joint limits for the Nao robot (for clamping)
                "LShoulderPitch": (-2.08567, 2.08567),
                "LShoulderRoll": (-0.314159, 1.32645),
                "LHipYawPitch": (-1.14529, 0.740718),
                "LHipRoll": (-0.379435, 0.79046),
                "LHipPitch": (-1.77378, 0.48398),
                "LKneePitch": (-0.0923279, 2.11255),
                "LAnklePitch": (-1.18944, 0.922581),
                "LAnkleRoll": (-0.39788, 0.769001),
                #######################################
                "RShoulderPitch": (-2.08567, 2.08567),
                "RShoulderRoll": (-1.32645, 0.314159),
                "RHipYawPitch": (-1.14529, 0.740718),
                "RHipRoll": (-0.738274, 0.449597),
                "RHipPitch": (-1.77378, 0.48398),
                "RKneePitch": (-0.0923279, 2.11255),
                "RAnklePitch": (-1.1863, 0.932006),
                "RAnkleRoll": (-0.768992, 0.397935)
               } """
JOINT_LIMITS = {      # restricted joint limits for the Nao robot
                "LShoulderPitch": (-2.0, 2.0),
                "LShoulderRoll":  (-0.3, 1.3),
                "LHipYawPitch":   (-0.5, 0.4),
                "LHipRoll":       (-0.2, 0.2),
                "LHipPitch":      (-0.8, 0.0),
                "LKneePitch":     (0.5, 1.5),
                "LAnklePitch":    (-0.9, 0.0),
                "LAnkleRoll":     (-0.2, 0.2),
                #######################################
                "RShoulderPitch": (-2.0, 2.0),
                "RShoulderRoll":  (-1.3, 0.3),
                "RHipYawPitch":   (-0.6, 0.4),
                "RHipRoll":       (-0.25, 0.2),
                "RHipPitch":      (-0.9, 0.0),
                "RKneePitch":     (0.6, 1.5),
                "RAnklePitch":    (-0.9, 0.0),
                "RAnkleRoll":     (-0.2, 0.2)
               }

if WANDB:
    wandb.init(
        project="nao_cga",
        config={"num_generations": NUM_GENERATIONS,
                "population_size": POPULATION_SIZE, 
                "mutation_rate":   MUTATION_RATE,
                "num_joints":      PARAMS,
                "num_activations": NUM_ACTIVATIONS}
    )

###########################################################################
## Initialize Supervisor and Devices
###########################################################################
robot = Supervisor()  ##used to control the simulation
gps = robot.getDevice("gps")  ##to get the position of the robot
gps.enable(TIME_STEP)

###########################################################################
## Motor Initialization
###########################################################################
motor_names = [
               #"LShoulderPitch",
               #"LShoulderRoll",
               #"LHipYawPitch",
               "LHipRoll",
               "LHipPitch",
               "LKneePitch",
               "LAnklePitch",
               "LAnkleRoll",
               ###############
               #"RShoulderPitch",
               #"RShoulderRoll",
               #"RHipYawPitch",
               "RHipRoll",
               "RHipPitch",
               "RKneePitch",
               "RAnklePitch",
               "RAnkleRoll"]

motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(0.0)  # set all motors to default position

###########################################################################
## Get the initial position and rotation of the robot
###########################################################################
root = robot.getRoot()  # get the root node of the robot
children_field = root.getField("children")  # get the children field of the root node
robot_node = next((children_field.getMFNode(i) for i in range(children_field.getCount())
                   if children_field.getMFNode(i).getTypeName() == "Nao"), None)
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
initial_position = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()

def get_joint_limits(): # use to find the limits for each motor
    for name in motor_names:
        curr_motor = robot.getDevice(name)
        min_position = curr_motor.getMinPosition()
        max_position = curr_motor.getMaxPosition()
        print(f"{name}: min_position={min_position}, max_position={max_position}")
    robot.step(TIME_STEP)  # Run a single simulation step to initialize devices

def reset_robot(): # Reset the robot to the initial state
    #robot.simulationResetPhysics()
    robot.simulationReset()
    for motor in motors:
        motor.setPosition(0.0)  # set all motors to default position
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)
    for _ in range(10): 
        robot.step(TIME_STEP)  # Step the simulation a few times to stabilize the reset

def clamp(value, min_value, max_value): # Clamp a value within a specific range
    return max(min(value, max_value), min_value)

def create_CPG_individual(): # Create an individual with random motor parameters
    return {"amplitude": [random.uniform(0, 0.5) for _ in range(PARAMS)],  ##amplitude of the sine wave
            "phase": [random.uniform(0, 2 * math.pi) for _ in range(PARAMS)],  ##phase of the sine wave
            "offset": [random.uniform(-0.5, 0.5) for _ in range(PARAMS)],  ##offset of the sine wave
            "fitness": 0.0}  ##fitness value of the individual

def create_cyclic_individual():
    return {"amplitude": [[random.uniform(0, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##amplitude of the sine wave
            "phase": [[random.uniform(0, 2 * math.pi) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##phase of the sine wave
            "offset": [[random.uniform(-0.5, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##offset of the sine wave
            "repetitions": [random.randint(10, 40) for _ in range(NUM_ACTIVATIONS)],  ##number of repetitions of the gait cycle
            "fitness": 0.0}  ##fitness value of the individual

def create_position_individual(): # Create an individual with random motor positions rather than sin wave
    individual = []
    for _ in range(1, 35):
        cycle = []
        for _ in range(len(motors)):
            position = random.uniform(-1, 1)
            position = round(position, 2)
            cycle.append(position)
        individual.append(cycle)
    return individual

def test_population():
    pop_size = 5
    population = [create_CPG_individual() for _ in range(pop_size)]
    #population = [create_cyclic_individual() for _ in range(pop_size)]
    for individual in population:
        print("----------------------------------------------------") 
        #print(individual)
        for i, motor in enumerate(motors):
            motor_name = motor.getName()
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                individual["amplitude"][i] = clamp(individual["amplitude"][i], min_limit, max_limit)
                individual["phase"][i] = clamp(individual["phase"][i], min_limit, max_limit)
                individual["offset"][i] = clamp(individual["offset"][i], min_limit, max_limit)
            print(f"Motor {i}: {motor.getName()} \n Amplitude: {individual['amplitude'][i]:.3f}, \n Phase: {individual['phase'][i]:.3f}, \n Offset: {individual['offset'][i]:.3f}")
        print("----------------------------------------------------")

def evaluate_OG(individual): # Evaluate fitness of an individual
    reset_robot() # Reset robot to the initial state before evaluating each individual
    start_time = robot.getTime()
    max_distance, height_sum, height_samples = 0.0, 0.0, 0
    initial_pos = gps.getValues()
    f = 0.75  # Gait frequency (1 Hz?)
    
    count = 0
    while robot.getTime() - start_time < 20.0:  ##Run the simulation for 20 seconds
        time = robot.getTime()
        for i, motor in enumerate(motors):  # iterate over all motors
            # calculate the position of the motor
            position = (individual["amplitude"][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][i]) + individual["offset"][i])
            motor_name = motor.getName()
            # Apply clamping based on joint-specific limits
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)
            motor.setPosition(position) # set the position of the motor to clamped value
        robot.step(TIME_STEP)
        count += 1
        print("count: ", count)
        current_pos = gps.getValues()
        distance = math.sqrt((current_pos[0] - initial_pos[0]) ** 2 + (current_pos[2] - initial_pos[2]) ** 2)
        max_distance = max(max_distance, distance)
        height_sum += current_pos[1]
        height_samples += 1
    avg_height = height_sum / height_samples if height_samples > 0 else 0.0
    #print("average height:", avg_height)
    #print("max distance:", max_distance)
    individual["fitness"] = max_distance + avg_height * HEIGHT_WEIGHT
    return individual["fitness"]

def evaluate(individual): # Evaluate fitness of an individual
    #print("individual: ", individual)
    reset_robot() # Reset robot to the initial state before evaluating each individual
    robot.getDevice("RShoulderPitch").setPosition(0.6)  # move right arm down
    robot.getDevice("LShoulderPitch").setPosition(0.6)  # move left arm down
    start_time = robot.getTime()
    initial_pos = gps.getValues()
    distance, total_forward_distance, height_sum, height_samples, height_bonus = 0.0, 0.0, 0.0, 0, 0.0
    prev_dist = 0
    f = 0.75  # Gait frequency (Hz?)

    count, current_activation = 0, 0
    while robot.getTime() - start_time < 10.0:  # Run the simulation for 20 seconds
        time = robot.getTime()
        for i, motor in enumerate(motors):  # iterate over all motors
            # motor position: y(t) = A * sin(2 * pi * f * t + phi) + C ## (AMPLITUDE * (sin (2 * pi * frequency * time + PHASE) + OFFSET))
            position = (individual["amplitude"][current_activation][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][current_activation][i]) + individual["offset"][current_activation][i])
            motor_name = motor.getName()
            if motor_name in JOINT_LIMITS: # value clamping based on joint limits
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)
            motor.setPosition(position) # set the position of the motor to clamped value
        robot.step(TIME_STEP)
        count += 1
        current_pos = gps.getValues()
        #print("current_pos: ", current_pos)
        #distance = math.sqrt((current_pos[0] - initial_pos[0]) ** 2 + (current_pos[1] - initial_pos[1]) ** 2)
        distance = current_pos[0] - initial_pos[0] # only x-axis (forward) distance
        forward_distance = current_pos[0] - prev_dist
        height = current_pos[2]
        #print(height)
        
        if height > 0.1:
            forward_distance = forward_distance * 1.5
            height_bonus += 0.0002
        elif height < 0.22:
            forward_distance = forward_distance * 2
            height_bonus += 0.0004
        elif height < 0.31:
            forward_distance = forward_distance * 2.5
            height_bonus += 0.0006
        
        if forward_distance > 0:
            total_forward_distance += forward_distance
        prev_dist = current_pos[0]

        height_sum += height
        height_samples += 1
        
        if count >= individual["repetitions"][current_activation]:
            count = 0
            current_activation += 1
            if current_activation > (NUM_ACTIVATIONS - 1):
                current_activation = 0
            #print("current_activation: ", current_activation, "-> reps: ", individual["repetitions"][current_activation])
    
    avg_height = height_sum / height_samples if height_samples > 0 else 0.0
    fitness = distance + total_forward_distance + height_bonus + (avg_height * HEIGHT_WEIGHT)
    #fitness = total_forward_distance + (avg_height * HEIGHT_WEIGHT)
    
    individual["fitness"] = fitness
    print("____     average height:", avg_height)
    print("____         height sum:", height_sum * .01)
    print("____       height bonus:", height_bonus)
    print("____     total distance:", distance)
    print("____   forward distance:", total_forward_distance)
    print("____            FITNESS:", fitness)
    return individual["fitness"]

def evaluate_positional(individual): # Evaluate fitness of an individual
    #print("individual: ", individual)
    reset_robot() # Reset robot to the initial state before evaluating each individual
    robot.getDevice("RShoulderPitch").setPosition(1)  # move right arm down
    robot.getDevice("LShoulderPitch").setPosition(1)  # move left arm down
    start_time = robot.getTime()
    initial_pos = gps.getValues()
    max_distance, total_distance, height_sum, height_samples = 0.0, 0.0, 0.0, 0
    f = 0.75  # Gait frequency (Hz?)

    count, current_activation = 0, 0
    while robot.getTime() - start_time < 20.0:  # Run the simulation for 20 seconds
        cycle_time = time.time()
        cycle_end_time = cycle_time + 0.07 # 40 milliseconds per cycle
        for i, motor in enumerate(motors):  # iterate over all motors
            # calculate the position of the motor
            position = (individual[current_activation][i])
            motor_name = motor.getName()
            # Apply clamping based on joint-specific limits
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)
            motor.setPosition(position) # set the position of the motor to clamped value
        robot.step(TIME_STEP)
        count += 1
        
        current_pos = gps.getValues()
        #distance = math.sqrt((current_pos[0] - initial_pos[0]) ** 2 + (current_pos[2] - initial_pos[2]) ** 2)
        distance = current_pos[0] - initial_pos[0] # only x-axis (forward) distance
        total_distance += distance
        max_distance = max(max_distance, distance)
        height_sum += current_pos[1]
        height_samples += 1

        #if count >= individual["repetitions"][current_activation]:
        if cycle_time > cycle_end_time:
            count = 0
            current_activation += 1
            if current_activation > (NUM_ACTIVATIONS - 1):
                current_activation = 0
            #print("current_activation: ", current_activation, "-> reps: ", individual["repetitions"][current_activation])

    avg_height = height_sum / height_samples if height_samples > 0 else 0.0
    fitness = max_distance + (avg_height * HEIGHT_WEIGHT)
    #print("average height: ", avg_height)
    #print("max distance: ", max_distance)
    #print("total distance: ", total_distance)
    #print("fitness: ", fitness)
    individual["fitness"] = fitness
    return individual["fitness"]

def mutate_OG(individual):
    for i in range(PARAMS):
        if random.random() < MUTATION_RATE:
            individual["amplitude"][i] += random.uniform(-0.05, 0.05)
            individual["phase"][i] += random.uniform(-0.05, 0.05)
            individual["offset"][i] += random.uniform(-0.05, 0.05)

def mutate(individual):
    for i in range(NUM_ACTIVATIONS):
        for j in range(PARAMS):
            if random.random() < MUTATION_RATE:
                individual["amplitude"][i][j] += random.uniform(-0.05, 0.05)
                individual["phase"][i][j] += random.uniform(-0.05, 0.05)
                individual["offset"][i][j] += random.uniform(-0.05, 0.05)
        if random.random() < MUTATION_RATE:
            individual["repetitions"][i] += random.randint(-5, 5)

def crossover_OG(parent1, parent2):
    child = create_individual()
    for i in range(PARAMS):
        if random.choice([True, False]):
            child["amplitude"][i] = parent1["amplitude"][i]
            child["phase"][i] = parent1["phase"][i]
            child["offset"][i] = parent1["offset"][i]
        else:
            child["amplitude"][i] = parent2["amplitude"][i]
            child["phase"][i] = parent2["phase"][i]
            child["offset"][i] = parent2["offset"][i]
    mutate(child)
    return child

def crossover(parent1, parent2):
    child = create_cyclic_individual()
    for i in range(NUM_ACTIVATIONS):
        if random.choice([True, False]):
            child["amplitude"][i] = parent1["amplitude"][i]
            child["phase"][i] = parent1["phase"][i]
            child["offset"][i] = parent1["offset"][i]
            child["repetitions"][i] = parent1["repetitions"][i]
        else:
            child["amplitude"][i] = parent2["amplitude"][i]
            child["phase"][i] = parent2["phase"][i]
            child["offset"][i] = parent2["offset"][i]
            child["repetitions"][i] = parent2["repetitions"][i]
    mutate(child)
    return child

def select_parent(population):
    total_fitness = sum(ind["fitness"] for ind in population)
    selection_probs = [ind["fitness"] / total_fitness for ind in population] if total_fitness > 0 else None
    return random.choices(population, weights=selection_probs, k=1)[0] if selection_probs else random.choice(population)

def evolve_population(population): # Evolutionary process to create a new generation
    population.sort(key=lambda ind: ind["fitness"], reverse=True)
    new_population = population[:25] # keep the top 20 individuals
    for ind in new_population:
        mutate(ind)
    while len(new_population) < POPULATION_SIZE - 3:
        parent1, parent2 = select_parent(population), select_parent(population)
        child = crossover(parent1, parent2)
        new_population.append(child)
    while len(new_population) < POPULATION_SIZE:
        new_population.append(create_cyclic_individual())
    return new_population

##########################################################################################
def run(): # Main Loop
    #population = [create_CPG_individual() for _ in range(POPULATION_SIZE)]
    #population = [create_position_individual() for _ in range(POPULATION_SIZE)]
    population = [create_cyclic_individual() for _ in range(POPULATION_SIZE)]
    best_individuals = []
    for gen in range(NUM_GENERATIONS):
        print(" ")
        print("#################################################")
        print(f"############## Generation {gen+1} ##############")
        print("#################################################")
        print(" ")
        for i, individual in enumerate(population):
            print(" ")
            print("------------------ Evaluating Individual", i+1, "------------------------")
            print(" ")
            individual["fitness"] = evaluate(individual)
            reset_robot()
        
        best_individual = max(population, key=lambda ind: ind["fitness"])
        best_individuals.append(best_individual)

        best_fitness = best_individual["fitness"]
        mean_fitness = sum(ind["fitness"] for ind in population) / POPULATION_SIZE
        print(f"Best Individual in Generation {gen}: {best_individual['fitness']:.3f}")
        if WANDB:
            wandb.log({"Best Fitness": best_fitness, "Mean Fitness": mean_fitness})
        population = evolve_population(population)
    print("############## Evolution Complete ##############")
    print("Best Individuals:")
    for i, best_individual in enumerate(best_individuals):
        print(f"Generation {i}: {best_individual['fitness']:.3f}")

def hardcoded(): # hardcoded gait cycle
    start_time = time.time()
    end_time = start_time + 20.0  # Run the simulation for 20 seconds
    while time.time() < end_time:
        cycle_time = time.time()
        cycle_end_time = cycle_time + 0.07 # 40 milliseconds per cycle
        while time.time() < cycle_end_time:
            for cyc in cycle.CYCLE:
                for j, motor in enumerate(motors):
                    motor.setPosition(cyc[j])
                robot.step(TIME_STEP)

##########################################################################################
if __name__ == "__main__":
    run()
    #test_population()
    #get_joint_limits()
    #hardcoded()