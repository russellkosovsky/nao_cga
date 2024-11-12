from controller import Robot, GPS, Supervisor
import math
import random

# Constants
NUM_GENERATIONS = 15
POPULATION_SIZE = 50
MUTATION_RATE = 0.03
PARAMS = 10
TIME_STEP = 20
HEIGHT_WEIGHT = 0.2

JOINT_LIMITS = {
    "RHipRoll": (-0.738274, 0.449597),
    "RHipPitch": (-1.77378, 0.48398),
    "RKneePitch": (-0.0923279, 2.11255),
    "RAnklePitch": (-1.1863, 0.932006),
    "RAnkleRoll": (-0.768992, 0.397935),
    "LHipRoll": (-0.379435, 0.79046),
    "LHipPitch": (-1.77378, 0.48398),
    "LKneePitch": (-0.0923279, 2.11255),
    "LAnklePitch": (-1.18944, 0.922581),
    "LAnkleRoll": (-0.39788, 0.769001)
}

# Initialize Supervisor and Devices
robot = Supervisor()
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Motor Initialization
motor_names = ["RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
               "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"]
motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(0.0)
    
robot.getDevice("RShoulderPitch").setPosition(1.1)
robot.getDevice("LShoulderPitch").setPosition(1.1)
    
# Get the initial position and rotation of the robot
root = robot.getRoot()
children_field = root.getField("children")
robot_node = next((children_field.getMFNode(i) for i in range(children_field.getCount())
                   if children_field.getMFNode(i).getTypeName() == "Nao"), None)
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
initial_position = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()   
 
def get_joint_limits():
    for name in motor_names:
        motor = robot.getDevice(name)
        min_position = motor.getMinPosition()
        max_position = motor.getMaxPosition()
        print(f"{name}: min_position={min_position}, max_position={max_position}")
    robot.step(TIME_STEP)  # Run a single simulation step to initialize devices


# Generate an individual with bounded knee constraints
def create_individual():
    return {
        "amplitude": [random.uniform(0, 0.5) for _ in range(PARAMS)],
        "phase": [random.uniform(0, 2 * math.pi) for _ in range(PARAMS)],
        "offset": [random.uniform(-0.5, 0.5) for _ in range(PARAMS)],
        "fitness": 0.0
    }


# Clamp values within a specific range
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


# Function to reset the robot to the initial state
def reset_robot():
    for motor in motors: # Reset motor positions
        motor.setPosition(0.0)
    # Reset the robot's position and orientation
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)
    for _ in range(3): # Step the simulation a few times to stabilize the reset
        robot.step(TIME_STEP)


# Evaluate fitness of an individual
def evaluate(individual):
    reset_robot() # Reset robot to the initial state before evaluating each individual
    start_time = robot.getTime()
    max_distance, height_sum, height_samples = 0.0, 0.0, 0
    initial_pos = gps.getValues()
    f = 1.0  # Gait frequency
    
    while robot.getTime() - start_time < 20.0:
        
        time = robot.getTime()
        for i, motor in enumerate(motors):
            position = (individual["amplitude"][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][i])
                        + individual["offset"][i])
            motor_name = motor.getName()

            # Apply clamping based on joint-specific limits
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)

            motor.setPosition(position)

        robot.step(TIME_STEP)
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


# Mutation
def mutate(individual):
    for i in range(PARAMS):
        if random.random() < MUTATION_RATE:
            individual["amplitude"][i] += random.uniform(-0.05, 0.05)
            individual["phase"][i] += random.uniform(-0.05, 0.05)
            individual["offset"][i] += random.uniform(-0.05, 0.05)


# Crossover between two parents to create a child
def crossover(parent1, parent2):
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


# Roulette wheel selection based on fitness
def select_parent(population):
    total_fitness = sum(ind["fitness"] for ind in population)
    selection_probs = [ind["fitness"] / total_fitness for ind in population] if total_fitness > 0 else None
    return random.choices(population, weights=selection_probs, k=1)[0] if selection_probs else random.choice(population)


# Evolutionary process to create a new generation
def evolve_population(population):
    population.sort(key=lambda ind: ind["fitness"], reverse=True)
    new_population = population[:POPULATION_SIZE // 2]
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = select_parent(population), select_parent(population)
        child = crossover(parent1, parent2)
        new_population.append(child)
    return new_population


# Main Evolution Loop
def main():
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    for gen in range(NUM_GENERATIONS):
        print(f"############## Generation {gen} ##############")
        for i, individual in enumerate(population):
            print("  Individual", i)
            individual["fitness"] = evaluate(individual)
            reset_robot()
            print(f"    Fitness: {individual['fitness']:.3f}")
        population = evolve_population(population)


if __name__ == "__main__":
    main()
