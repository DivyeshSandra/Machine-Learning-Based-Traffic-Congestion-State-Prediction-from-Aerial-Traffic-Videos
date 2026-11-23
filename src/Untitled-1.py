
#for performing roulette wheel selection, crossover and mutation.
import matplotlib.pyplot as plt

#initial population as per the question.
population = ["1100","0100","0001","1110","0111","1001"]

#fitness function.
def fitness(binary):
    x = int(binary, 2)  #converting binary to decimal
    return 15 * x - x**2

#function to display individuals and fitness.
def evaluate_population(population):

    fitness_values = []
    print("\nPopulation Evaluation")
    print("Individual  Binary  Decimal  Fitness")
    
    for i, chromosome in enumerate(population):

        decimal = int(chromosome, 2)
        fit = fitness(chromosome)

        fitness_values.append(fit)

        print(f"S{i+1:<10} {chromosome:<8} "f"{decimal:<9} {fit}")

    print("Total Fitness:", sum(fitness_values))

    return fitness_values


#function to draw roulette wheel.
def draw_roulette(population, fitness_values, title):

    total_fitness = sum(fitness_values)

    probabilities = [value / total_fitness for value in fitness_values]

    labels = [f"S{i+1}: {population[i]}" for i in range(len(population))]

    print("\nRoulette Wheel Probabilities")

    for i, probability in enumerate(probabilities):
        print(f"S{i+1}: {probability*100:.2f}%")

    plt.figure(figsize=(8, 7))
    plt.pie(
        fitness_values,
        labels=labels,
        autopct="%1.2f%%",
        startangle=90
    )

    plt.title(title)
    plt.axis("equal")
    plt.show()

#initial fitness and roulette wheel.
print("(a)Initial population")

initial_fitness = evaluate_population(population)

draw_roulette(
    population,
    initial_fitness,
    "Initial Roulette Wheel"
)

initial_total = sum(initial_fitness)

#(b) one point crossover at 3rd position between s2 and s6.
print("\n(b)One Point Crossover.")

parent1 = population[5]  # S6 = 1001
parent2 = population[1]  # S2 = 0100

cut_point = 3

child1 = parent1[:cut_point] + parent2[cut_point:]
child2 = parent2[:cut_point] + parent1[cut_point:]

print("Parent S6:", parent1)
print("Parent S2:", parent2)

print("Child 1:", child1)
print("Child 2:", child2)

#Replacing corresponding parents
population[5] = child1
population[1] = child2

print("\nPopulation After Crossover:")
for i, chromosome in enumerate(population):
    print(f"S{i+1}: {chromosome}")

#mutation of S1 and S5 at positions 2 and 3.
print("\n(c)Mutation")

def mutate(chromosome, position):

    bits = list(chromosome)
    index = position - 1 #conversion into python index
    bits[index] = '1' if bits[index] == '0' else '0' # Flip the selected bit

    return ''.join(bits)


#Mutate S1 at positions 2 and 3
population[0] = mutate(population[0], 2)
population[0] = mutate(population[0], 3)

#Mutate S5 at positions 2 and 3
population[4] = mutate(population[4], 2)
population[4] = mutate(population[4], 3)

print("After Mutation:")

for i, chromosome in enumerate(population):
    print(f"S{i+1}: {chromosome}")

#(D) Final fitness, roulette wheel and improvement percentage.

print("\nPART D: FINAL POPULATION")

final_fitness = evaluate_population(population)

final_total = sum(final_fitness)

draw_roulette(
    population,
    final_fitness,
    "Final Roulette Wheel"
)


#Calculation of improvement percentage
improvement = (
    (final_total - initial_total)
    / initial_total
) * 100

print("\nFITNESS IMPROVEMENT")

print("Initial Total Fitness:", initial_total)
print("Final Total Fitness:", final_total)

print(
    f"Fitness Improvement: {improvement:.2f}%"
)
