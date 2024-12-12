###########################################################################
import math
import random
import time
import wandb
import cycle 
from controller import ROBOT, GPS, Supervisor
###########################################################################
## Constants
###########################################################################
#WANDB = False
WANDB = True
NUM_GENERATIONS = 400
POPULATION_SIZE = 200
MUTATION_RATE = 0.003
PARAMS = 10           # number of controlled MOTORS
NUM_ACTIVATIONS = 15  # number of actions (gait cycles per individual)
TIME_STEP = 60        # default time step
HEIGHT_WEIGHT = 8    # weight for the height component of the fitness
"""JOINT_LIMITS = {      # joint limits for the Nao ROBOT (for clamping)
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
JOINT_LIMITS = {      # restricted joint limits for the Nao ROBOT
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
ROBOT = Supervisor()  ##used to control the simulation
GPS = ROBOT.getDevice("GPS")  ##to get the position of the ROBOT
GPS.enable(TIME_STEP)

###########################################################################
## Motor Initialization
###########################################################################
MOTOR_NAMES = [
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

MOTORS = [ROBOT.getDevice(name) for name in MOTOR_NAMES]
for motor in MOTORS:
    motor.setPosition(0.0)  # set all MOTORS to default position

###########################################################################
## Get the initial position and rotation of the ROBOT
###########################################################################
ROOT = ROBOT.getROOT()  # get the ROOT node of the ROBOT
CHILD_FIELD = ROOT.getField("children")  # get the children field of the ROOT node
ROBOT_NODE = next((CHILD_FIELD.getMFNode(i) for i in range(CHILD_FIELD.getCount())
                   if CHILD_FIELD.getMFNode(i).getTypeName() == "Nao"), None)
TRANSLATION_FIELD = ROBOT_NODE.getField("translation")
ROTATION_FIELD = ROBOT_NODE.getField("rotation")
INITIAL_POSITION = TRANSLATION_FIELD.getSFVec3f()
INITIAL_ROTATION = ROTATION_FIELD.getSFRotation()


def get_joint_limits(): # use to find the limits for each motor
    for name in MOTOR_NAMES:
        curr_motor = ROBOT.getDevice(name)
        min_position = curr_motor.getMinPosition()
        max_position = curr_motor.getMaxPosition()
        print(f"{name}: min_position={min_position}, max_position={max_position}")
    ROBOT.step(TIME_STEP)  # Run a single simulation step to initialize devices

def reset_robot(): # Reset the ROBOT to the initial state
    #ROBOT.simulationResetPhysics()
    ROBOT.simulationReset()
    for motor in MOTORS:
        motor.setPosition(0.0)  # set all MOTORS to default position
    TRANSLATION_FIELD.setSFVec3f(INITIAL_POSITION)
    ROTATION_FIELD.setSFRotation(INITIAL_ROTATION)
    for _ in range(10): 
        ROBOT.step(TIME_STEP)  # Step the simulation a few times to stabilize the reset

def clamp(value, min_value, max_value): # Clamp a value within a specific range
    return max(min(value, max_value), min_value)


def create_cyclic_individual():
    return {"amplitude": [[random.uniform(0, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##amplitude of the sine wave
            "phase": [[random.uniform(0, 2 * math.pi) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##phase of the sine wave
            "offset": [[random.uniform(-0.5, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##offset of the sine wave
            "repetitions": [random.randint(10, 40) for _ in range(NUM_ACTIVATIONS)],  ##number of repetitions of the gait cycle
            "fitness": 0.0}  ##fitness value of the individual

class Individual: 
    def __init__(self, amplitude, phase, offset, repetitions, fitness):
        self.amplitude = amplitude
        self.phase = phase
        self.offset = offset
        self.repetitions = repetitions
        self.fitness = fitness

    def __str__(self):
        return f"Amplitude: {self.amplitude}, Phase: {self.phase}, Offset: {self.offset}, Repetitions: {self.repetitions}, Fitness: {self.fitness}"
    
    def create_individual(self):
        self.amplitude= [[random.uniform(0, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##amplitude of the sine wave
        self.phase = [[random.uniform(0, 2 * math.pi) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##phase of the sine wave
        self.offset = [[random.uniform(-0.5, 0.5) for _ in range(PARAMS)] for _ in range(NUM_ACTIVATIONS)],  ##offset of the sine wave
        self.repetitions = [random.randint(10, 40) for _ in range(NUM_ACTIVATIONS)],  ##number of repetitions of the gait cycle
        self.fitness = 0.0  ##fitness value of the individual


def mutate(individual):
    for i in range(NUM_ACTIVATIONS):
        for j in range(PARAMS):
            if random.random() < MUTATION_RATE:
                individual["amplitude"][i][j] += random.uniform(-0.05, 0.05)
                individual["phase"][i][j] += random.uniform(-0.05, 0.05)
                individual["offset"][i][j] += random.uniform(-0.05, 0.05)
        if random.random() < MUTATION_RATE:
            individual["repetitions"][i] += random.randint(-5, 5)

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
    new_population = population[:3] # keep the top 3 individuals
    for ind in new_population:
        mutate(ind)
    while len(new_population) < POPULATION_SIZE - 5:
        parent1, parent2 = select_parent(population), select_parent(population)
        child = crossover(parent1, parent2)
        new_population.append(child)
    while len(new_population) < POPULATION_SIZE:
        new_population.append(create_cyclic_individual())
    return new_population

class population:
    def __init__(self, size):
        self.size = size
        self.population = [Individual() for _ in range(size)]
    
    def __str__(self):
        return f"Population size: {self.size}, Population: {self.population}"

    def create_population(self):
        self.population = [Individual() for _ in range(self.size)]

    def evaluate_population(self):
        for i, individual in enumerate(self.population):
            print("____")
            print("____ Evaluating Individual", i+1, " ------------------------------")
            print("____")
            individual.fitness = evaluate(individual)
            reset_robot()

    def get_best_individual(self):
        return max(self.population, key=lambda ind: ind.fitness)

    def evolve_population(self):
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        new_population = self.population[:3] # keep the top 3 individuals
        for ind in new_population:
            mutate(ind)
        while len(new_population) < POPULATION_SIZE - 5:
            parent1, parent2 = select_parent(self.population), select_parent(self.population)
            child = crossover(parent1, parent2)
            new_population.append(child)
        while len(new_population) < POPULATION_SIZE:
            new_population.append(create_cyclic_individual())
        self.population = new_population


def evaluate(individual): # Evaluate fitness of an individual
    #print("individual: ", individual)
    reset_robot() # Reset ROBOT to the initial state before evaluating each individual
    ROBOT.getDevice("RShoulderPitch").setPosition(1.5)  # move right arm down
    ROBOT.getDevice("LShoulderPitch").setPosition(1.5)  # move left arm down
    start_time = ROBOT.getTime()
    initial_pos = GPS.getValues()
    distance, total_forward_distance, height_sum, height_samples, height_bonus = 0.0, 0.0, 0.0, 0, 0.0
    prev_dist = 0
    f = 0.75  # Gait frequency (Hz?)

    count, current_activation = 0, 0
    while ROBOT.getTime() - start_time < 20.0:  # Run the simulation for 20 seconds
        time = ROBOT.getTime()
        for i, motor in enumerate(MOTORS):  # iterate over all MOTORS
            # motor position: y(t) = A * sin(2 * pi * f * t + phi) + C ## (AMPLITUDE * (sin (2 * pi * frequency * time + PHASE) + OFFSET))
            position = (individual["amplitude"][current_activation][i] * math.sin(2.0 * math.pi * f * time + individual["phase"][current_activation][i]) + individual["offset"][current_activation][i])
            motor_name = motor.getName()
            if motor_name in JOINT_LIMITS: # value clamping based on joint limits
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                position = clamp(position, min_limit, max_limit)
            motor.setPosition(position) # set the position of the motor to clamped value
        ROBOT.step(TIME_STEP)
        count += 1
        current_pos = GPS.getValues()
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
    
    #fitness = distance + total_forward_distance + height_bonus + (avg_height * HEIGHT_WEIGHT)
    fitness = total_forward_distance + height_bonus + (avg_height * HEIGHT_WEIGHT)

    #fitness = total_forward_distance + (avg_height * HEIGHT_WEIGHT)
    
    individual["fitness"] = fitness
    print("____     average height:", avg_height)
    print("____         height sum:", height_sum * .01)
    print("____       height bonus:", height_bonus)
    print("____     total distance:", distance)
    print("____   forward distance:", total_forward_distance)
    print("____            FITNESS:", round(fitness, 2))
    return individual["fitness"]


##########################################################################################
##########################################################################################

def run_OG(): # Main Loop
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
            print("____")
            print("____ Evaluating Individual", i+1, " ------------------------------")
            print("____")
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


def run(): # Main Loop using population and individiual classes
    pop = population(POPULATION_SIZE)
    pop.create_population()
    for gen in range(NUM_GENERATIONS):
        print(" ")
        print("#################################################")
        print(f"############## Generation {gen+1} ##############")
        print("#################################################")
        print(" ")
        pop.evaluate_population()
        best_individual = pop.get_best_individual()
        print(f"Best Individual in Generation {gen}: {best_individual.fitness:.3f}")
        if WANDB:
            wandb.log({"Best Fitness": best_individual.fitness})
        pop.evolve_population()
    print("############## Evolution Complete ##############")
    best_individual = pop.get_best_individual()
    print(f"Best Individual: {best_individual.fitness:.3f}")

##########################################################################################
def hardcoded(): # hardcoded gait cycle
    start_time = time.time()
    end_time = start_time + 20.0  # Run the simulation for 20 seconds
    while time.time() < end_time:
        cycle_time = time.time()
        cycle_end_time = cycle_time + 0.07 # 40 milliseconds per cycle
        while time.time() < cycle_end_time:
            for cyc in cycle.CYCLE:
                for j, motor in enumerate(MOTORS):
                    motor.setPosition(cyc[j])
                ROBOT.step(TIME_STEP)

def test_population():
    pop_size = 5
    population = [create_cyclic_individual() for _ in range(pop_size)]
    for individual in population:
        print("----------------------------------------------------") 
        #print(individual)
        for i, motor in enumerate(MOTORS):
            motor_name = motor.getName()
            if motor_name in JOINT_LIMITS:
                min_limit, max_limit = JOINT_LIMITS[motor_name]
                individual["amplitude"][i] = clamp(individual["amplitude"][i], min_limit, max_limit)
                individual["phase"][i] = clamp(individual["phase"][i], min_limit, max_limit)
                individual["offset"][i] = clamp(individual["offset"][i], min_limit, max_limit)
            print(f"Motor {i}: {motor.getName()} \n Amplitude: {individual['amplitude'][i]:.3f}, \n Phase: {individual['phase'][i]:.3f}, \n Offset: {individual['offset'][i]:.3f}")
        print("----------------------------------------------------")

##########################################################################################
if __name__ == "__main__":
    run_OG()
    run()
    #test_population()
    #get_joint_limits()
    #hardcoded()