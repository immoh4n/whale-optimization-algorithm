class WOA:
    def __init__(self, max_iter=200, population_size=20, a=0.5, b=1):
        self.max_iter = max_iter
        self.population_size = population_size
        self.a = a
        self.b = b

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        # Initialize population with all features selected
        self.population = np.ones((self.population_size, data.shape[1]))
        
        for epoch in range(self.max_iter):
            # Update population
            self.update_population()

        # Select best solution
        best_fitness = min(self.evaluate_fitness())
        best_index = np.argmin(self.evaluate_fitness())
        self.best_solution = self.population[best_index]
        self.best_features = self.data.columns[self.best_solution.astype(bool)].tolist()

    def update_population(self):
        for i in range(self.population_size):
            r = np.random.rand()
            A = 2 * self.a * r - self.a
            C = 2 * r
            p = np.random.rand()

            if p < 0.5:
                if np.abs(A) < 1:
                    self.population[i] = self.search_preys(A, C, i)
                else:
                    rand_leader_index = np.random.randint(0, self.population_size)
                    rand_leader = self.population[rand_leader_index]
                    self.population[i] = rand_leader + A * (rand_leader - self.population[i])
            else:
                distance_to_leader = np.abs(self.population[i] - self.population[0])
                self.population[i] = distance_to_leader * np.exp(self.b * C) * np.cos(2 * np.pi * C) + self.population[0]

    def evaluate_fitness(self):
        fitness = []
        for solution in self.population:
            selected_features = self.data.columns[solution.astype(bool)].tolist()
            # Here, you would apply your liver disease prediction model to evaluate the fitness of each solution
            # For demonstration purposes, let's assume a simple fitness based on the difference between predicted and actual labels
            # Random prediction for demonstration
            predicted_labels = np.random.randint(0, 2, size=len(self.labels))  
            fitness.append(np.abs(predicted_labels - self.labels).sum())
        return fitness
        

    def search_preys(self, A, C, i):
        # Ensure that the solution remains unchanged
        return self.population[i]